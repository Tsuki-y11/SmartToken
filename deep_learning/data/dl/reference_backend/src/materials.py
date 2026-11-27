"""Bounded in-memory extraction of untrusted, base64-supplied materials.

Nothing here opens user paths, fetches URLs, evaluates formulae, or persists
source bytes. ``context_text`` is a three-line envelope whose middle line is
JSON; integrations must pass it as source data, never planner instructions.
See docs/coverage-calibration/materials-report.md for the public contract.
"""
import base64
import binascii
import csv
import hashlib
import io
import json
import os
import posixpath
import re
import shutil
import subprocess
import sys
import time
import warnings
import zipfile
import zlib

from .contracts import ContractError


MAX_ATTACHMENTS = 4
MAX_FILE_BYTES = 4 * 1024 * 1024
MAX_TOTAL_BYTES = 8 * 1024 * 1024
MAX_FILE_TEXT_CHARS = 100_000
MAX_TOTAL_TEXT_CHARS = 100_000
MAX_CONTEXT_CHARS = 3_000_000
MAX_SEGMENTS = 10_000
MAX_TOTAL_SEGMENTS = 20_000
MAX_ARCHIVE_ENTRIES = 1_000
MAX_ARCHIVE_BYTES = 20 * 1024 * 1024
MAX_ARCHIVE_RATIO = 100
MAX_XML_NODES = 100_000
MAX_JSON_DEPTH = 128
MAX_PDF_PAGES = 100
MAX_PDF_DECODED_BYTES = 20 * 1024 * 1024
MAX_PDF_RESOURCE_DEPTH = 64
MAX_PDF_RESOURCE_OBJECTS = 10_000
MAX_PDF_FORM_INVOCATIONS = 512
MAX_PDF_EXPANDED_CONTENT_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 16_000_000
MAX_IMAGE_DIMENSION = 10_000
OCR_TIMEOUT_SECONDS = 15
OCR_REQUEST_BUDGET_SECONDS = 60
MAX_PDF_OCR_PAGES = 8
MAX_PDF_OCR_PAGE_PIXELS = 3_000_000
MAX_PDF_OCR_TOTAL_PIXELS = 16_000_000
MAX_PDF_OCR_DIMENSION = 3_000
MAX_PDF_OCR_SOURCE_PIXELS = 64_000_000
PDF_RENDER_TIMEOUT_SECONDS = 15

_TYPES = {
    'txt': 'text/plain', 'md': 'text/markdown', 'csv': 'text/csv',
    'json': 'application/json', 'pdf': 'application/pdf',
    'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    'xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    'png': 'image/png', 'jpg': 'image/jpeg', 'jpeg': 'image/jpeg',
}
_PNG = b'\x89PNG\r\n\x1a\n'
_JPEG = b'\xff\xd8\xff'
_ZIP = b'PK\x03\x04'
_OLE = b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1'
_W = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
_M = '{http://schemas.openxmlformats.org/officeDocument/2006/math}'
_S = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'
_R = '{http://schemas.openxmlformats.org/officeDocument/2006/relationships}'


def _fail(code, message):
    raise ContractError(message, code)


def _bound(condition, message):
    if not condition:
        _fail('MATERIAL_TOO_LARGE', message)


def _validate_attachment(item):
    if not isinstance(item, dict) or not {'name', 'content_base64'} <= item.keys() or set(item) - {'name', 'content_base64', 'mime_type'}:
        _fail('MATERIAL_INVALID', 'Each material needs only name, content_base64, and optional mime_type.')
    name = item['name']
    if not isinstance(name, str) or not 1 <= len(name) <= 255 or name != name.strip() or name in ('.', '..') or any(c in name for c in '/\\:') or any(ord(c) < 32 or ord(c) == 127 for c in name):
        _fail('MATERIAL_INVALID', 'Material name must be a plain filename, not a path or URL.')
    extension = name.rsplit('.', 1)[-1].lower() if '.' in name else ''
    if extension not in _TYPES:
        _fail('MATERIAL_UNSUPPORTED', 'Supported material formats: TXT, MD, CSV, JSON, PDF, DOCX, XLSX, PNG, JPEG.')
    hint = item.get('mime_type')
    if hint is not None and (not isinstance(hint, str) or not 1 <= len(hint) <= 127 or any(ord(c) < 32 or ord(c) > 126 for c in hint)):
        _fail('MATERIAL_INVALID', 'mime_type must be a short printable string.')
    encoded = item['content_base64']
    if not isinstance(encoded, str) or not encoded:
        _fail('MATERIAL_INVALID_BASE64', 'Material contents must use canonical base64.')
    _bound(len(encoded) <= 4 * ((MAX_FILE_BYTES + 2) // 3), 'Encoded material exceeds the per-file byte limit.')
    try:
        data = base64.b64decode(encoded, validate=True)
        if base64.b64encode(data).decode('ascii') != encoded:
            raise ValueError('noncanonical')
    except (binascii.Error, ValueError):
        _fail('MATERIAL_INVALID_BASE64', 'Material contents must use canonical base64.')
    _bound(len(data) <= MAX_FILE_BYTES, 'Decoded material exceeds the per-file byte limit.')
    if extension in ('docx', 'xlsx') and data.startswith(_OLE):
        _fail('MATERIAL_ENCRYPTED', 'Encrypted Office containers and legacy Office formats are unsupported.')
    signature = {'pdf': b'%PDF-', 'docx': _ZIP, 'xlsx': _ZIP, 'png': _PNG, 'jpg': _JPEG, 'jpeg': _JPEG}.get(extension)
    if signature and not data.startswith(signature):
        _fail('MATERIAL_CORRUPT', 'Material bytes do not match the declared filename format.')
    return name, extension, hint, data


class _Extraction:
    def __init__(self):
        self.segments = []
        self.text_chars = 0
        self.warnings = []
        self.requires_review = False
        self.method = ''
        self.ocr_languages = []
        self.text_coverage = None

    def add(self, text, reference):
        _bound(len(self.segments) < MAX_SEGMENTS, 'Material has too many text locations.')
        self.text_chars += len(text)
        _bound(self.text_chars <= MAX_FILE_TEXT_CHARS, 'Extracted material text exceeds the per-file limit.')
        self.segments.append(dict(reference=reference, text=text))

    def warn(self, code, review=True):
        if code not in self.warnings:
            self.warnings.append(code)
        self.requires_review = self.requires_review or review


class _OCRRequestBudget:
    """One deadline shared by every PDF/image OCR process in this request."""
    def __init__(self):
        self.deadline = time.monotonic() + OCR_REQUEST_BUDGET_SECONDS
        self.exhausted = False

    def check(self):
        remaining = self.deadline - time.monotonic()
        if self.exhausted or remaining <= 0:
            self.exhausted = True
            _fail('MATERIAL_OCR_REQUEST_BUDGET_EXCEEDED', 'The shared OCR request time budget was exhausted.')
        return remaining

    def run(self, command, *, timeout, **options):
        remaining = self.check()
        effective_timeout = min(timeout, remaining)
        try:
            return subprocess.run(command, timeout=effective_timeout, **options)
        except subprocess.TimeoutExpired:
            # subprocess.run kills and waits for the timed-out child. Do not
            # launch background work or allow another source to reset time.
            if remaining <= timeout or time.monotonic() >= self.deadline:
                self.exhausted = True
                self.check()
            raise


def _decode_text(data):
    if data.startswith((_PNG, _JPEG, _ZIP, _OLE, b'%PDF-')):
        _fail('MATERIAL_CORRUPT', 'Binary document signature found in a text material.')
    text = data.decode('utf-16' if data.startswith((b'\xff\xfe', b'\xfe\xff')) else 'utf-8-sig')
    if any(ord(c) < 32 and c not in '\t\n\r' for c in text) or '\x7f' in text:
        _fail('MATERIAL_CORRUPT', 'Text materials must contain UTF-8 or BOM-marked UTF-16 text without binary control characters.')
    _bound(len(text) <= MAX_FILE_TEXT_CHARS, 'Decoded material text exceeds the per-file limit.')
    return text


def _reject_json_constant(value):
    raise ValueError('Non-finite JSON value')


def _unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON object key')
        result[key] = value
    return result


def _validate_json(text):
    depth, quoted, escaped = 0, False, False
    for character in text:
        if quoted:
            if escaped:
                escaped = False
            elif character == '\\':
                escaped = True
            elif character == '"':
                quoted = False
        elif character == '"':
            quoted = True
        elif character in '[{':
            depth += 1
            _bound(depth <= MAX_JSON_DEPTH, 'JSON nesting exceeds the depth limit.')
        elif character in ']}':
            depth -= 1
    json.loads(text, parse_constant=_reject_json_constant, object_pairs_hook=_unique_json_object)


def _column_name(index):
    name = ''
    while index:
        index, remainder = divmod(index - 1, 26)
        name = chr(65 + remainder) + name
    return name


def _text(data, extension, result):
    text = _decode_text(data)
    encoding = 'utf16-bom' if data.startswith((b'\xff\xfe', b'\xfe\xff')) else 'utf8'
    result.method = encoding + '-text'
    if extension == 'json':
        _validate_json(text)
        result.method = 'json-validated-' + encoding
    if extension == 'csv':
        result.method = 'csv-reader'
        reader = csv.reader(io.StringIO(text, newline=''), strict=True)
        previous_line = 0
        for row_number, row in enumerate(reader, 1):
            line_start = previous_line + 1
            previous_line = reader.line_num
            for column, value in enumerate(row, 1):
                cell = _column_name(column) + str(row_number)
                result.add(value, dict(kind='csv_cell', ref='cell ' + cell, cell=cell, row=row_number,
                                       column=column, line_start=line_start, line_end=reader.line_num))
    else:
        for line_number, line in enumerate(text.splitlines(), 1):
            result.add(line, dict(kind='line', ref=f'line {line_number}', line_start=line_number, line_end=line_number))


def _archive(data):
    """Check declared limits before inflating, then validate actual bytes/CRC."""
    package = zipfile.ZipFile(io.BytesIO(data))
    try:
        entries = package.infolist()
        _bound(len(entries) <= MAX_ARCHIVE_ENTRIES, 'Document archive has too many entries.')
        total = 0
        names = set()
        for entry in entries:
            name = entry.filename
            if not name or name in names or '\\' in name or ':' in name or name.startswith('/') or any(p in ('.', '..') for p in name.split('/')) or any(ord(c) < 32 for c in name):
                _fail('MATERIAL_UNSAFE_ARCHIVE', 'Unsafe or duplicate document archive entry.')
            names.add(name)
            if entry.flag_bits & 1:
                _fail('MATERIAL_ENCRYPTED', 'Encrypted document archive entries are unsupported.')
            if entry.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
                _fail('MATERIAL_UNSUPPORTED', 'Document archive compression is unsupported.')
            total += entry.file_size
            _bound(total <= MAX_ARCHIVE_BYTES, 'Document archive exceeds its uncompressed byte limit.')
            _bound(entry.file_size <= MAX_ARCHIVE_RATIO * max(entry.compress_size, 1), 'Document archive compression ratio exceeds its limit.')
            if 'vbaproject' in name.lower() or '/embeddings/' in name.lower():
                _fail('MATERIAL_UNSUPPORTED', 'Macros and embedded document packages are unsupported.')
        contents = {}
        for entry in entries:
            with package.open(entry) as member:
                content = member.read(entry.file_size + 1)
            if len(content) != entry.file_size:
                _fail('MATERIAL_CORRUPT', 'Document archive entry length is inconsistent.')
            contents[entry.filename] = content
        return contents
    finally:
        package.close()


def _xml(data):
    try:
        from defusedxml import ElementTree
    except ImportError:
        _fail('MATERIAL_DEPENDENCY_UNAVAILABLE', 'Office material extraction requires defusedxml.')
    root = ElementTree.fromstring(data, forbid_dtd=True, forbid_entities=True, forbid_external=True)
    _bound(sum(1 for _ in root.iter()) <= MAX_XML_NODES, 'Document XML exceeds the node limit.')
    return root


def _office(data, extension, result):
    contents = _archive(data)
    if '[Content_Types].xml' not in contents:
        _fail('MATERIAL_CORRUPT', 'Office document package manifest is missing.')
    manifest = _xml(contents['[Content_Types].xml'])
    content_types = [node.get('ContentType', '') for node in manifest]
    required = ('application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml' if extension == 'docx'
                else 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml')
    if required not in content_types:
        _fail('MATERIAL_CORRUPT', 'Office package content type does not match its filename.')
    for name, content in contents.items():
        if name.endswith('.rels'):
            for rel in _xml(content):
                if rel.get('TargetMode', '').lower() == 'external':
                    _fail('MATERIAL_UNSAFE_ARCHIVE', 'External Office relationships are unsupported and are never fetched.')
    result.method = 'ooxml-defusedxml'
    if extension == 'docx':
        _docx(contents, result)
    else:
        _xlsx(contents, result)


def _docx(contents, result):
    if 'word/document.xml' not in contents:
        _fail('MATERIAL_CORRUPT', 'DOCX main document part is missing.')
    parts = ['word/document.xml'] + sorted(n for n in contents if re.fullmatch(r'word/(?:header\d+|footer\d+|footnotes|endnotes|comments)\.xml', n))
    for part in parts:
        root = _xml(contents[part])
        if any(node.tag in (_W + 'drawing', _W + 'pict', _W + 'object', _W + 'altChunk') for node in root.iter()):
            result.warn('DOCX_NON_TEXT_CONTENT_UNVERIFIED')
        if any(node.tag in (_W + 'instrText', _W + 'fldSimple', _W + 'del') for node in root.iter()):
            result.warn('DOCX_FIELDS_OR_REVISIONS_UNVERIFIED')
        if any(node.tag.startswith(_M) for node in root.iter()):
            result.warn('DOCX_EQUATION_LAYOUT_UNVERIFIED')
        for number, paragraph in enumerate(root.iter(_W + 'p'), 1):
            pieces = []
            for node in paragraph.iter():
                if node.tag in (_W + 't', _M + 't'):
                    pieces.append(node.text or '')
                elif node.tag == _W + 'tab':
                    pieces.append('\t')
                elif node.tag in (_W + 'br', _W + 'cr'):
                    pieces.append('\n')
            result.add(''.join(pieces), dict(kind='paragraph', ref=f'{part}, paragraph {number}', part=part, paragraph=number))
    if any(n.startswith('word/media/') for n in contents):
        result.warn('DOCX_NON_TEXT_CONTENT_UNVERIFIED')


def _xlsx_displayed_string(item):
    """OOXML phonetic runs are annotations, not displayed cell text."""
    pieces = []
    if item is not None:
        for node in item:
            if node.tag == _S + 't':
                pieces.append(node.text or '')
            elif node.tag == _S + 'r':
                pieces.extend(text.text or '' for text in node.findall(_S + 't'))
    return ''.join(pieces)


def _xlsx(contents, result):
    if 'xl/workbook.xml' not in contents or 'xl/_rels/workbook.xml.rels' not in contents:
        _fail('MATERIAL_CORRUPT', 'XLSX workbook or relationships part is missing.')
    strings = []
    if 'xl/sharedStrings.xml' in contents:
        for item in _xml(contents['xl/sharedStrings.xml']).iter(_S + 'si'):
            strings.append(_xlsx_displayed_string(item))
    relationships = {}
    for rel in _xml(contents['xl/_rels/workbook.xml.rels']):
        target = rel.get('Target', '')
        if '\\' in target or ':' in target or any(p == '..' for p in target.split('/')):
            _fail('MATERIAL_UNSAFE_ARCHIVE', 'Unsafe workbook relationship target.')
        target = target.lstrip('/') if target.startswith('/') else posixpath.normpath(posixpath.join('xl', target))
        relationships[rel.get('Id')] = (target, rel.get('Type', ''))
    workbook = _xml(contents['xl/workbook.xml'])
    sheet_names = set()
    for sheet in workbook.iter(_S + 'sheet'):
        name = sheet.get('name', '')
        if not name or name in sheet_names:
            _fail('MATERIAL_CORRUPT', 'Invalid or duplicate worksheet name.')
        sheet_names.add(name)
        target, rel_type = relationships.get(sheet.get(_R + 'id'), ('', ''))
        if target not in contents or not rel_type.endswith('/worksheet'):
            _fail('MATERIAL_CORRUPT', 'Worksheet part or relationship is missing.')
        root = _xml(contents[target])
        seen_cells = set()
        for cell in root.iter(_S + 'c'):
            address = cell.get('r', '')
            match = re.fullmatch(r'([A-Z]{1,3})([1-9][0-9]{0,6})', address)
            if not match or address in seen_cells or int(match[2]) > 1048576:
                _fail('MATERIAL_CORRUPT', 'Invalid or duplicate worksheet cell address.')
            column = 0
            for letter in match[1]:
                column = column * 26 + ord(letter) - 64
            if column > 16384:
                _fail('MATERIAL_CORRUPT', 'Worksheet column exceeds the XLSX limit.')
            seen_cells.add(address)
            value_node = cell.find(_S + 'v')
            value = value_node.text or '' if value_node is not None else ''
            value_type = cell.get('t', 'n')
            if value_type == 's':
                if not value.isdigit() or int(value) >= len(strings):
                    _fail('MATERIAL_CORRUPT', 'Invalid shared string index.')
                value = strings[int(value)]
            elif value_type == 'inlineStr':
                value = _xlsx_displayed_string(cell.find(_S + 'is'))
            elif value_type not in ('n', 'str', 'b', 'e', 'd'):
                _fail('MATERIAL_CORRUPT', 'Unsupported worksheet cell value type.')
            reference = dict(kind='cell', ref=f'{name}!{address}', sheet=name, cell=address, stored_type=value_type)
            formula = cell.find(_S + 'f')
            if formula is not None:
                result.warn('XLSX_FORMULA_CACHE_UNVERIFIED')
                reference['formula'] = True
                reference['cache_status'] = 'unverified' if value_node is not None and value != '' else 'missing'
                if reference['cache_status'] == 'missing':
                    result.warn('XLSX_FORMULA_CACHE_MISSING')
                    value = '[cached value unavailable]'
                value = f'formula (not executed): {formula.text or "[shared/array formula]"}; cached value (unverified): {value}'
            if cell.get('s') is not None:
                reference['style_index'] = cell.get('s')
                result.warn('XLSX_FORMATTED_VALUES_UNVERIFIED')
            if value_type == 'e':
                result.warn('XLSX_ERROR_CELL_PRESENT')
            result.add(value, reference)
        if any(node.tag in (_S + 'drawing', _S + 'legacyDrawing', _S + 'oleObjects') for node in root.iter()):
            result.warn('XLSX_NON_TEXT_CONTENT_UNVERIFIED')
    if not sheet_names:
        _fail('MATERIAL_CORRUPT', 'XLSX workbook has no worksheets.')


def _pdf_has_inline_image(data):
    """Find a BI operator without parsing or decoding inline image pixels.

    Input has already passed the decoded stream cap. Skip literal/hex strings,
    comments, and names so ordinary quoted BI text is not a visual-content flag.
    """
    delimiters = b'\x00\t\n\f\r ()<>[]{}/%'
    offset, length = 0, len(data)
    while offset < length:
        character = data[offset]
        if character == 37:  # comment
            while offset < length and data[offset] not in (10, 13):
                offset += 1
        elif character == 40:  # nested literal string
            offset, depth = offset + 1, 1
            while offset < length and depth:
                if data[offset] == 92:
                    offset += 2
                    continue
                depth += (data[offset] == 40) - (data[offset] == 41)
                offset += 1
        elif character == 60 and data[offset:offset + 2] != b'<<':
            end = data.find(b'>', offset + 1)
            offset = end + 1 if end >= 0 else length
        elif character == 47:  # name object
            offset += 1
            while offset < length and data[offset] not in delimiters:
                offset += 1
        elif data[offset:offset + 2] == b'<<':
            offset += 2
        elif character in delimiters:
            offset += 1
        else:
            start = offset
            while offset < length and data[offset] not in delimiters:
                offset += 1
            if data[start:offset] == b'BI':
                return True
    return False


class _PDFStreamBudget:
    """Preflight reachable text/resources before pypdf recursively extracts forms.

    The budget is document-local. Flate output is capped during decompression;
    the resulting cache prevents pypdf from running an unbounded second decode.
    Image streams are not decoded because this path only extracts PDF text.
    """
    def __init__(self):
        self.decoded_bytes = 0
        self.seen = set()
        self.active = set()
        self.stream_data = {}
        self.form_work = {}
        self.scanned_content = set()
        self.has_inline_images = False
        self.has_image_xobjects = False
        self.raster_images_safe = True
        self.source_image_pixels = 0

    def visit(self, value, depth=0):
        from pypdf.generic import StreamObject
        if hasattr(value, 'get_object'):
            value = value.get_object()
        if not isinstance(value, (dict, list)):
            return
        identity = id(value)
        if identity in self.active:
            _fail('MATERIAL_CORRUPT', 'Cyclic PDF text resources are unsupported.')
        if identity in self.seen:
            return
        _bound(depth <= MAX_PDF_RESOURCE_DEPTH, 'PDF resource nesting exceeds the limit.')
        _bound(len(self.seen) + len(self.active) < MAX_PDF_RESOURCE_OBJECTS, 'PDF resource object count exceeds the limit.')
        if isinstance(value, StreamObject) and value.get('/Subtype') == '/Image':
            self.has_image_xobjects = True
            width, height = value.get('/Width', 0), value.get('/Height', 0)
            if (not isinstance(width, (int, float)) or not isinstance(height, (int, float))
                    or not 0 < width <= MAX_IMAGE_DIMENSION or not 0 < height <= MAX_IMAGE_DIMENSION
                    or width * height > MAX_IMAGE_PIXELS):
                self.raster_images_safe = False
            else:
                self.source_image_pixels += width * height
                if self.source_image_pixels > MAX_PDF_OCR_SOURCE_PIXELS:
                    self.raster_images_safe = False
            self.seen.add(identity)
            return
        self.active.add(identity)
        if isinstance(value, StreamObject):
            self._decode(value)
        children = value.values() if isinstance(value, dict) else value
        for child in children:
            self.visit(child, depth + 1)
        self.active.remove(identity)
        self.seen.add(identity)

    def _decode(self, stream):
        from pypdf.generic import DecodedStreamObject, EncodedStreamObject, NullObject
        raw = stream._data
        filters = stream.get('/Filter')
        if hasattr(filters, 'get_object'):
            filters = filters.get_object()
        if isinstance(filters, list):
            if len(filters) != 1:
                _fail('MATERIAL_UNSUPPORTED', 'PDF text stream filter chains are unsupported.')
            filters = filters[0].get_object()
        remaining = MAX_PDF_DECODED_BYTES - self.decoded_bytes
        if filters is None:
            decoded = raw
        elif filters in ('/FlateDecode', '/Fl'):
            parameters = stream.get('/DecodeParms')
            if hasattr(parameters, 'get_object'):
                parameters = parameters.get_object()
            if isinstance(parameters, list):
                if len(parameters) != 1:
                    _fail('MATERIAL_UNSUPPORTED', 'PDF text stream decode parameters are unsupported.')
                parameters = parameters[0].get_object()
            if parameters is not None and not isinstance(parameters, NullObject):
                if not isinstance(parameters, dict) or parameters.get('/Predictor', 1) != 1:
                    _fail('MATERIAL_UNSUPPORTED', 'Predictor-encoded PDF text streams are unsupported.')
            decoder = zlib.decompressobj()
            decoded = decoder.decompress(raw, remaining + 1)
            _bound(len(decoded) <= remaining, 'PDF decoded text/resource content exceeds the byte limit.')
            if not decoder.eof or decoder.unused_data:
                _fail('MATERIAL_CORRUPT', 'PDF text stream compression is incomplete or ambiguous.')
        else:
            _fail('MATERIAL_UNSUPPORTED', 'PDF text stream compression is unsupported.')
        _bound(len(decoded) <= remaining, 'PDF decoded text/resource content exceeds the byte limit.')
        self.decoded_bytes += len(decoded)
        self.stream_data[id(stream)] = decoded
        if isinstance(stream, EncodedStreamObject):
            cached = DecodedStreamObject()
            cached.set_data(decoded)
            stream.decoded_self = cached

    @staticmethod
    def _resolve(value):
        return value.get_object() if hasattr(value, 'get_object') else value

    def _resources(self, owner):
        resources = self._resolve(owner.get_inherited('/Resources', {}))
        if resources is None:
            return {}
        if not isinstance(resources, dict):
            _fail('MATERIAL_CORRUPT', 'PDF resources must be a dictionary.')
        return resources

    def _content_stats(self, value):
        """Count repeated array streams without joining or reparsing their data."""
        value = self._resolve(value)
        if value is None:
            return 0, 0
        streams = value if isinstance(value, list) else [value]
        _bound(len(streams) <= MAX_PDF_RESOURCE_OBJECTS, 'PDF content array exceeds the work limit.')
        byte_count, possible_calls = 0, 0
        for stream in streams:
            stream = self._resolve(stream)
            data = self.stream_data.get(id(stream))
            if data is None:
                _fail('MATERIAL_CORRUPT', 'PDF content stream was not safely decoded.')
            if id(stream) not in self.scanned_content:
                self.has_inline_images = self.has_inline_images or _pdf_has_inline_image(data)
                self.scanned_content.add(id(stream))
            byte_count += len(data) + (1 if isinstance(value, list) else 0)
            _bound(byte_count <= MAX_PDF_EXPANDED_CONTENT_BYTES, 'PDF expanded content exceeds the work limit.')
            # Every real Do operator contains these bytes. Counting occurrences
            # inside comments/strings too is deliberately conservative and
            # avoids parsing attacker-controlled operators to establish a cap.
            possible_calls += data.count(b'Do')
        return byte_count, possible_calls

    def _resource_bytes(self, resources):
        """Include font/CMap and other local resources on every invocation."""
        pending, seen, size = [resources], set(), 0
        while pending:
            value = self._resolve(pending.pop())
            if not isinstance(value, (dict, list)) or id(value) in seen:
                continue
            seen.add(id(value))
            if isinstance(value, dict):
                if value.get('/Subtype') in ('/Form', '/Image'):
                    continue
                size += len(self.stream_data.get(id(value), b''))
                pending.extend(child for key, child in value.items() if key != '/XObject')
            else:
                pending.extend(value)
        return size

    def _work(self, owner, content, *, form=False):
        identity = id(owner)
        if form and identity in self.form_work:
            return self.form_work[identity]
        own_bytes, possible_calls = self._content_stats(content)
        resources = self._resources(owner)
        expanded_bytes = own_bytes + self._resource_bytes(resources)
        invocations = 1 if form else 0
        if possible_calls:
            xobjects = self._resolve(resources.get('/XObject', {}))
            if not isinstance(xobjects, dict):
                _fail('MATERIAL_CORRUPT', 'PDF XObject resources must be a dictionary.')
            max_calls, max_bytes = 0, 0
            for candidate in xobjects.values():
                candidate = self._resolve(candidate)
                if not isinstance(candidate, dict):
                    _fail('MATERIAL_CORRUPT', 'Invalid PDF XObject resource.')
                if candidate.get('/Subtype') == '/Image':
                    continue
                child_calls, child_bytes = self._work(candidate, candidate, form=True)
                max_calls = max(max_calls, child_calls)
                max_bytes = max(max_bytes, child_bytes)
            # Each possible invocation is charged for the most expensive
            # eligible resource; aliases and repeated Do tokens are not free.
            invocations += possible_calls * max_calls
            expanded_bytes += possible_calls * max_bytes
        _bound(invocations <= MAX_PDF_FORM_INVOCATIONS, 'PDF possible Form invocations exceed the work limit.')
        _bound(expanded_bytes <= MAX_PDF_EXPANDED_CONTENT_BYTES, 'PDF expanded content/resources exceed the work limit.')
        if form:
            self.form_work[identity] = (invocations, expanded_bytes)
        return invocations, expanded_bytes

    def check_work(self, pages):
        invocations, expanded_bytes = 0, 0
        for page in pages:
            calls, size = self._work(page, page.get('/Contents'))
            invocations += calls
            expanded_bytes += size
            _bound(invocations <= MAX_PDF_FORM_INVOCATIONS, 'PDF aggregate Form invocations exceed the work limit.')
            _bound(expanded_bytes <= MAX_PDF_EXPANDED_CONTENT_BYTES, 'PDF aggregate expanded work exceeds the byte limit.')


def _pdf(data, result, ocr_budget):
    try:
        from pypdf import PdfReader
    except ImportError:
        _fail('MATERIAL_DEPENDENCY_UNAVAILABLE', 'PDF extraction requires pypdf.')
    reader = PdfReader(io.BytesIO(data), strict=True)
    if reader.is_encrypted:
        _fail('MATERIAL_ENCRYPTED', 'Encrypted PDFs are unsupported.')
    _bound(len(reader.pages) <= MAX_PDF_PAGES, 'PDF page count exceeds the limit.')
    result.method = 'pypdf-text'
    budget = _PDFStreamBudget()
    for page in reader.pages:
        budget.visit(page.get('/Contents'))
        budget.visit(page.get('/Resources'))
    budget.check_work(reader.pages)
    if budget.has_inline_images or budget.has_image_xobjects:
        result.warn('PDF_NON_TEXT_CONTENT_UNVERIFIED')
    ocr = _PDFPageOCR(data, result, budget.raster_images_safe, ocr_budget)
    native_pages, ocr_pages, missing_pages = 0, 0, 0
    for page_number, page in enumerate(reader.pages, 1):
        if page.get('/Annots'):
            result.warn('PDF_ANNOTATIONS_UNVERIFIED')
        text = page.extract_text() or ''
        reference = dict(kind='page', ref=f'page {page_number}', page=page_number)
        if text.strip():
            native_pages += 1
            reference['extraction_status'] = 'native_text'
        else:
            text, status = ocr.read(page_number - 1)
            reference['extraction_status'] = status
            if text.strip():
                ocr_pages += 1
        if not text.strip():
            missing_pages += 1
            result.warn('PDF_PAGE_WITHOUT_TEXT')
        result.add(text, reference)
    result.text_coverage = dict(unit='page', total=len(reader.pages), with_text=native_pages + ocr_pages,
                               native_text=native_pages, ocr_text=ocr_pages, without_text=missing_pages,
                               status='partial' if missing_pages else 'all_pages_have_text')


class _PDFPageOCR:
    """Optional OCR never discards readable native text on another page.

    PDFium executes in a short-lived subprocess. The source/resource preflight
    has completed before this object is used; rendering does not replace it.
    """
    def __init__(self, data, result, raster_images_safe, ocr_budget):
        self.data, self.result = data, result
        self.ocr_budget = ocr_budget
        self.raster_images_safe = raster_images_safe
        self.pages, self.pixels = 0, 0
        self.executable, self.languages = None, []
        self.unavailable = False

    def _missing(self, warning, status):
        self.result.warn(warning)
        return '', status

    def read(self, page_index):
        try:
            self.ocr_budget.check()
        except ContractError:
            return self._missing('OCR_REQUEST_BUDGET_EXCEEDED', 'ocr_request_budget_exceeded')
        if self.pages >= MAX_PDF_OCR_PAGES:
            return self._missing('PDF_OCR_PAGE_LIMIT', 'ocr_page_limit')
        remaining = MAX_PDF_OCR_TOTAL_PIXELS - self.pixels
        if remaining <= 0 or not self.raster_images_safe:
            return self._missing('PDF_OCR_PIXEL_LIMIT', 'ocr_pixel_limit')
        if self.unavailable:
            return self._missing('PDF_OCR_UNAVAILABLE', 'ocr_unavailable')
        self.pages += 1
        try:
            if self.executable is None:
                self.executable = shutil.which('tesseract')
                if not self.executable:
                    _fail('MATERIAL_DEPENDENCY_UNAVAILABLE', 'Local Tesseract is unavailable.')
            if not self.languages:
                self.languages = _ocr_languages(self.executable, self.ocr_budget)
            limit = min(remaining, MAX_PDF_OCR_PAGE_PIXELS)
            helper = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_pdf_raster.py')
            completed = self.ocr_budget.run([sys.executable, helper, str(page_index), str(limit), str(MAX_PDF_OCR_DIMENSION)],
                                           input=self.data, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                           timeout=PDF_RENDER_TIMEOUT_SECONDS, check=False)
            if completed.returncode == 2:
                _fail('MATERIAL_DEPENDENCY_UNAVAILABLE', 'PDF OCR requires pypdfium2 and Pillow.')
            if completed.returncode == 3:
                return self._missing('PDF_OCR_PIXEL_LIMIT', 'ocr_pixel_limit')
            raster = completed.stdout
            if (completed.returncode or len(raster) > limit * 4 + 65_536 or len(raster) < 24
                    or not raster.startswith(_PNG) or raster[12:16] != b'IHDR'):
                _fail('MATERIAL_OCR_FAILED', 'PDF rasterization did not produce a bounded PNG.')
            width = int.from_bytes(raster[16:20], 'big')
            height = int.from_bytes(raster[20:24], 'big')
            if not 0 < width <= MAX_PDF_OCR_DIMENSION or not 0 < height <= MAX_PDF_OCR_DIMENSION or width * height > limit:
                return self._missing('PDF_OCR_PIXEL_LIMIT', 'ocr_pixel_limit')
            self.pixels += width * height
            text = _ocr_text(raster, self.executable, self.languages, self.ocr_budget, page_segmentation='3')
        except subprocess.TimeoutExpired:
            return self._missing('PDF_OCR_TIMEOUT', 'ocr_timeout')
        except OSError:
            self.unavailable = True
            return self._missing('PDF_OCR_UNAVAILABLE', 'ocr_unavailable')
        except ContractError as error:
            if error.code == 'MATERIAL_OCR_REQUEST_BUDGET_EXCEEDED':
                return self._missing('OCR_REQUEST_BUDGET_EXCEEDED', 'ocr_request_budget_exceeded')
            if error.code == 'MATERIAL_TOO_LARGE':
                raise
            if error.code == 'MATERIAL_DEPENDENCY_UNAVAILABLE':
                self.unavailable = True
                return self._missing('PDF_OCR_UNAVAILABLE', 'ocr_unavailable')
            if error.code == 'MATERIAL_OCR_TIMEOUT':
                return self._missing('PDF_OCR_TIMEOUT', 'ocr_timeout')
            return self._missing('PDF_OCR_FAILED', 'ocr_failed')
        self.result.method = 'pypdf-text+tesseract-ocr'
        self.result.ocr_languages = self.languages
        self.result.warn('OCR_TEXT_UNVERIFIED')
        self.result.warn('OCR_LANGUAGE_ENGLISH_ONLY' if self.languages == ['eng'] else 'OCR_LANGUAGE_COVERAGE_UNVERIFIED')
        self.result.warn('PDF_VISUAL_LAYOUT_UNVERIFIED')
        return (text, 'ocr_text_unverified') if text.strip() else ('', 'ocr_empty')


def _ocr_languages(executable, ocr_budget):
    configured = os.environ.get('SMARTTOKEN_OCR_LANGUAGES', 'eng')
    requested = configured.split('+')
    if (len(configured) > 128 or not 1 <= len(requested) <= 8 or len(set(requested)) != len(requested)
            or any(not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]{0,31}', language) for language in requested)):
        _fail('MATERIAL_DEPENDENCY_UNAVAILABLE', 'SMARTTOKEN_OCR_LANGUAGES must contain distinct language tokens joined with +.')
    try:
        installed = ocr_budget.run([executable, '--list-langs'], stdout=subprocess.PIPE,
                                   stderr=subprocess.DEVNULL, timeout=5, check=False)
    except subprocess.TimeoutExpired:
        _fail('MATERIAL_OCR_TIMEOUT', 'Local OCR language verification exceeded its time limit.')
    except OSError:
        _fail('MATERIAL_DEPENDENCY_UNAVAILABLE', 'Local Tesseract could not be started.')
    if installed.returncode or len(installed.stdout) > 65_536:
        _fail('MATERIAL_DEPENDENCY_UNAVAILABLE', 'Local Tesseract language data could not be verified.')
    try:
        available = set(installed.stdout.decode('utf-8').splitlines())
    except UnicodeDecodeError:
        _fail('MATERIAL_DEPENDENCY_UNAVAILABLE', 'Local Tesseract language listing is invalid.')
    missing = [language for language in requested if language not in available]
    if missing:
        _fail('MATERIAL_DEPENDENCY_UNAVAILABLE', 'Local OCR language data missing: ' + ', '.join(missing) + '.')
    return requested


def _ocr_text(png_data, executable, languages, ocr_budget, *, page_segmentation='6'):
    try:
        completed = ocr_budget.run([executable, 'stdin', 'stdout', '--psm', page_segmentation, '-l', '+'.join(languages)],
                                  input=png_data, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                  timeout=OCR_TIMEOUT_SECONDS, check=False)
    except subprocess.TimeoutExpired:
        _fail('MATERIAL_OCR_TIMEOUT', 'Local OCR exceeded its time limit.')
    except OSError:
        _fail('MATERIAL_DEPENDENCY_UNAVAILABLE', 'Local Tesseract could not be started.')
    if completed.returncode:
        _fail('MATERIAL_OCR_FAILED', 'Local OCR failed to extract the supplied image.')
    _bound(len(completed.stdout) <= MAX_FILE_TEXT_CHARS * 4, 'OCR output exceeds its byte limit.')
    try:
        return completed.stdout.decode('utf-8')
    except UnicodeDecodeError:
        _fail('MATERIAL_OCR_FAILED', 'Local OCR returned invalid text encoding.')


def _image(data, extension, result, ocr_budget):
    try:
        from PIL import Image
    except ImportError:
        _fail('MATERIAL_DEPENDENCY_UNAVAILABLE', 'Image extraction requires Pillow and local Tesseract.')
    with warnings.catch_warnings():
        warnings.simplefilter('error', Image.DecompressionBombWarning)
        with Image.open(io.BytesIO(data)) as picture:
            expected = 'PNG' if extension == 'png' else 'JPEG'
            if picture.format != expected or getattr(picture, 'n_frames', 1) != 1:
                _fail('MATERIAL_UNSUPPORTED', 'Image format mismatch or multi-frame image is unsupported.')
            width, height = picture.size
            _bound(width * height <= MAX_IMAGE_PIXELS and max(width, height) <= MAX_IMAGE_DIMENSION,
                   'Decoded image dimensions exceed the pixel limit.')
            picture.verify()
        with Image.open(io.BytesIO(data)) as picture:
            picture.load()
            buffer = io.BytesIO()
            picture.convert('RGB').save(buffer, format='PNG')
    result.method = 'tesseract-ocr'
    try:
        ocr_budget.check()
        executable = shutil.which('tesseract')
        if not executable:
            _fail('MATERIAL_DEPENDENCY_UNAVAILABLE', 'Local Tesseract executable with configured language data is required.')
        languages = _ocr_languages(executable, ocr_budget)
        text = _ocr_text(buffer.getvalue(), executable, languages, ocr_budget)
    except ContractError as error:
        if error.code != 'MATERIAL_OCR_REQUEST_BUDGET_EXCEEDED':
            raise
        result.warn('OCR_REQUEST_BUDGET_EXCEEDED')
        result.warn('IMAGE_VISUAL_UNDERSTANDING_UNVERIFIED')
        result.add('', dict(kind='image', ref='image', width=width, height=height,
                            extraction_status='ocr_request_budget_exceeded'))
        return
    result.ocr_languages = languages
    result.warn('OCR_TEXT_UNVERIFIED')
    result.warn('OCR_LANGUAGE_ENGLISH_ONLY' if languages == ['eng'] else 'OCR_LANGUAGE_COVERAGE_UNVERIFIED')
    result.warn('IMAGE_VISUAL_UNDERSTANDING_UNVERIFIED')
    for number, line in enumerate(text.splitlines(), 1):
        result.add(line, dict(kind='image_line', ref=f'image OCR line {number}', line_start=number,
                              line_end=number, width=width, height=height))


def extract_materials(attachments):
    """Extract bounded attachments; any invalid item rejects the whole request."""
    if not isinstance(attachments, list):
        _fail('MATERIAL_INVALID', 'attachments must be a list.')
    _bound(len(attachments) <= MAX_ATTACHMENTS, 'Too many material attachments.')
    if not attachments:
        return dict(context_text='', sources=[], warnings=[], requires_review=False, extracted_text_chars=0)
    ocr_budget = _OCRRequestBudget()
    prepared = []
    total_bytes = 0
    for item in attachments:
        entry = _validate_attachment(item)
        total_bytes += len(entry[3])
        _bound(total_bytes <= MAX_TOTAL_BYTES, 'Combined material bytes exceed the request limit.')
        prepared.append(entry)
    sources, context_sources, all_warnings = [], [], []
    total_text, total_segments = 0, 0
    requires_review = False
    has_actual_text = False
    for number, (name, extension, hint, data) in enumerate(prepared, 1):
        result = _Extraction()
        try:
            if extension in ('txt', 'md', 'json', 'csv'):
                _text(data, extension, result)
            elif extension in ('docx', 'xlsx'):
                _office(data, extension, result)
            elif extension == 'pdf':
                _pdf(data, result, ocr_budget)
            else:
                _image(data, extension, result, ocr_budget)
        except ContractError:
            raise
        except Exception:
            # Do not disclose parser exception strings: they can contain source text.
            raise ContractError('Material is corrupt or cannot be safely extracted.', 'MATERIAL_CORRUPT') from None
        source_has_text = any(segment['text'].strip() for segment in result.segments)
        has_actual_text = has_actual_text or source_has_text
        if not source_has_text and 'OCR_REQUEST_BUDGET_EXCEEDED' not in result.warnings:
            if extension == 'pdf':
                reason = (' Local OCR dependencies or configured language data are unavailable.'
                          if 'PDF_OCR_UNAVAILABLE' in result.warnings else
                          ' Local OCR timed out.' if 'PDF_OCR_TIMEOUT' in result.warnings else
                          ' OCR safety limits were reached.' if any(w in result.warnings for w in ('PDF_OCR_PAGE_LIMIT', 'PDF_OCR_PIXEL_LIMIT')) else
                          ' OCR did not return readable text.')
                _fail('MATERIAL_EMPTY_TEXT', 'PDF contains no extractable text after bounded OCR.' + reason +
                      ' Supply a searchable PDF or a verified text transcription.')
            _fail('MATERIAL_EMPTY_TEXT', 'Material contains no extractable text; supply readable source text or a verified OCR transcription.')
        if hint and hint.split(';', 1)[0].strip().lower() not in (_TYPES[extension], 'application/octet-stream'):
            result.warn('MATERIAL_MIME_MISMATCH')
        total_text += result.text_chars
        total_segments += len(result.segments)
        _bound(total_text <= MAX_TOTAL_TEXT_CHARS, 'Combined extracted text exceeds the request limit.')
        _bound(total_segments <= MAX_TOTAL_SEGMENTS, 'Combined text locations exceed the request limit.')
        source_id = f'M{number}'
        sha256 = hashlib.sha256(data).hexdigest()
        sources.append(dict(source_id=source_id, name=name, sha256=sha256, size_bytes=len(data),
                            media_type=_TYPES[extension], extraction_method=result.method,
                            text_chars=result.text_chars, references=[s['reference'] for s in result.segments],
                            warnings=result.warnings, requires_review=result.requires_review))
        if result.ocr_languages:
            sources[-1]['ocr_languages'] = result.ocr_languages
        if result.text_coverage is not None:
            sources[-1]['text_coverage'] = result.text_coverage
        context_sources.append(dict(source_id=source_id, name=name, sha256=sha256, segments=result.segments))
        for warning in result.warnings:
            if warning not in all_warnings:
                all_warnings.append(warning)
        requires_review = requires_review or result.requires_review
    if not has_actual_text:
        _fail('MATERIAL_EMPTY_TEXT', f'Materials contain no extractable text after the shared {OCR_REQUEST_BUDGET_SECONDS}-second OCR request budget was exhausted. '
              'Supply a searchable PDF, readable source text, or a verified OCR transcription.')
    body = json.dumps(dict(sources=context_sources), ensure_ascii=True, separators=(',', ':'))
    context_text = 'BEGIN_UNTRUSTED_MATERIALS_JSON\n' + body + '\nEND_UNTRUSTED_MATERIALS_JSON'
    _bound(len(context_text) <= MAX_CONTEXT_CHARS, 'Rendered material context exceeds the request limit.')
    return dict(context_text=context_text, sources=sources, warnings=all_warnings, requires_review=requires_review,
                extracted_text_chars=total_text)
