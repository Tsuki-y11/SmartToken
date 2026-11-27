"""Single-task local Ollama execution for the existing SYH Runner."""
import json
from pathlib import Path
import re
from time import perf_counter
from urllib.parse import urlsplit
from urllib.request import Request

from deep_learning.planner.hybrid.llm import _http_transport


def messages_for_prompt(prompt):
    source = (Path(__file__).resolve().parents[2] / 'server/prompts.js').read_text(encoding='utf-8')
    match = re.search(r"EXECUTION_SYSTEM_PROMPT\s*=\s*'([^']*)'", source)
    if not match:
        raise ValueError('LLC execution system prompt is missing')
    return [dict(role='system', content=match[1]), dict(role='user', content=prompt)]


def base_prompt(task, original_prompt):
    text = task['instruction']
    if 'original_prompt' in task['input_refs'] and text.strip() != original_prompt.strip():
        text = '原始用户请求：\n' + original_prompt + '\n\n当前子任务：\n' + text
    return text


_TASK_OUTPUT_PLACEHOLDER = re.compile(r'\{\{\s*(T[0-9]+\.output)\s*\}\}')


def build_messages(task, context):
    text = base_prompt(task, context.get('original_prompt', ''))
    placeholders = set(_TASK_OUTPUT_PLACEHOLDER.findall(text))
    if not placeholders.issubset(task['input_refs']):
        raise ValueError('Undeclared task output reference')
    output_refs = [ref for ref in task['input_refs'] if ref.endswith('.output')]
    for ref in output_refs:
        if ref not in context:
            raise ValueError('Missing upstream output')
    # 只替换原始模板，避免把上游输出再次解释为模板。
    text = _TASK_OUTPUT_PLACEHOLDER.sub(lambda match: context[match[1]], text)
    for ref in output_refs:
        if ref not in placeholders:
            text += f'\n\n{ref[:-7]} 实际输出：\n{context[ref]}'
    if _TASK_OUTPUT_PLACEHOLDER.search(text):
        raise ValueError('Unresolved task placeholder')
    return messages_for_prompt(text)


class OllamaAdapter:
    def __init__(self, base_url, deployment_model, tokenizer, *, model_id='qwen3-8b',
                 context_window=8192, transport=_http_transport):
        url = urlsplit(base_url)
        if url.scheme != 'http' or url.hostname not in ('127.0.0.1', 'localhost', '::1'):
            raise ValueError('Local Ollama endpoint required')
        if url.username or url.password or url.query or url.fragment or url.path not in ('', '/'):
            raise ValueError('Expected local Ollama origin')
        self.base_url = base_url.rstrip('/')
        self.deployment_model, self.model_id = deployment_model, model_id
        self.tokenizer, self.context_window, self.transport = tokenizer, context_window, transport

    def count_input_tokens(self, model_id, messages):
        if model_id != self.model_id:
            raise ValueError('Unknown model ID')
        return len(self.tokenizer.apply_chat_template(messages, tokenize=True,
                   add_generation_prompt=True, enable_thinking=False, return_dict=False))

    def call_llm(self, *, model_id, messages, generation, request_id, subtask_id):
        if model_id != self.model_id:
            raise ValueError('Unknown model ID')
        started = perf_counter()
        usage_in, usage_out, text = None, None, None
        failure = None
        try:
            data = dict(model=self.deployment_model, messages=messages, stream=False, think=False,
                        options=dict(temperature=generation['temperature'],
                                     num_predict=generation['max_output_tokens'], num_ctx=self.context_window))
            req = Request(self.base_url + '/api/chat', data=json.dumps(data).encode(),
                          headers={'Content-Type': 'application/json'})
            with self.transport(req, timeout=180) as response:
                document = json.loads(response.read(4 * 1024 * 1024))
            usage_in, usage_out = document.get('prompt_eval_count'), document.get('eval_count')
            text = document.get('message', {}).get('content')
            if document.get('model') != self.deployment_model:
                usage_in, usage_out = None, None
                raise ValueError('Provider model mismatch')
            if not document.get('done') or document.get('done_reason') == 'length':
                failure = dict(code='OUTPUT_TRUNCATED', message='Model did not finish within output limit')
            elif not isinstance(text, str) or not text.strip():
                failure = dict(code='MODEL_API_ERROR', message='Model returned no answer')
        except Exception as exc:
            failure = dict(code='MODEL_API_ERROR', message=f'Local model call failed ({type(exc).__name__})')
        known = all(type(n) is int and n >= 0 for n in (usage_in, usage_out))
        return dict(schema_version='1.0', request_id=request_id, subtask_id=subtask_id, model_id=model_id,
                    text=text if isinstance(text, str) else None,
                    input_tokens=usage_in if known else None, output_tokens=usage_out if known else None,
                    usage_source='provider' if known else 'unavailable',
                    latency_ms=round((perf_counter() - started) * 1000),
                    status='failed' if failure else 'succeeded', error=failure)
