"""V1.0 validation. Catalog remains the specified bare list, not a new envelope."""
import math

DIMENSIONS = {'agents', 'coding', 'reasoning', 'knowledge', 'math',
              'multilingual', 'multimodal', 'instruction_following'}


class ContractError(ValueError):
    def __init__(self, message, code='INVALID_REQUEST'):
        super().__init__(message)
        self.code = code


def require(condition, message, code='INVALID_REQUEST'):
    if not condition:
        raise ContractError(message, code)


def number(value):
    # JSON integers can exceed the floating-point range used by API numbers.
    # Bad upstream values must fail validation, never escape the billing path.
    try:
        return type(value) in (int, float) and math.isfinite(value) and value >= 0
    except (OverflowError, TypeError, ValueError):
        return False


def tokens(value):
    return type(value) is int and value >= 0


def envelope(obj, request_id):
    require(isinstance(obj, dict), 'expected JSON object')
    require(obj.get('schema_version') == '1.0', 'schema_version must be 1.0')
    require(obj.get('request_id') == request_id, 'request_id mismatch')


def validate_request(request):
    require(isinstance(request, dict), 'request must be an object')
    require(isinstance(request.get('request_id'), str) and bool(request['request_id']), 'missing request_id')
    envelope(request, request['request_id'])
    require(isinstance(request.get('original_prompt'), str) and bool(request['original_prompt'].strip()), 'empty prompt')
    c, g = request.get('constraints', {}), request.get('generation', {})
    require(isinstance(c, dict) and isinstance(g, dict), 'constraints and generation must be objects')
    require(number(c.get('budget_usd')), 'invalid budget')
    require(c.get('quality_priority') in ('cost_first', 'balanced', 'quality_first'), 'invalid quality_priority')
    require(type(c.get('max_subtasks')) is int and 1 <= c['max_subtasks'] <= 4, 'max_subtasks must be 1..4')
    require(type(c.get('allow_split')) is bool, 'allow_split must be bool')
    require(number(g.get('temperature')), 'invalid temperature')
    require(tokens(g.get('max_output_tokens')) and g['max_output_tokens'] > 0, 'invalid max_output_tokens')


def topological(tasks):
    remaining = {t['subtask_id']: t for t in tasks}
    require(len(remaining) == len(tasks), 'duplicate task IDs', 'INVALID_DAG')
    done, ordered = set(), []
    while remaining:
        ready = sorted(k for k, t in remaining.items() if set(t['depends_on']) <= done)
        require(bool(ready), 'cyclic DAG or missing dependency', 'INVALID_DAG')
        for key in ready:
            ordered.append(remaining.pop(key))
            done.add(key)
    return ordered


def validate_models(models):
    """Validate the unchanged bare catalog before any potentially paid work."""
    try:
        require(isinstance(models, list), 'models must be a list')
        seen = set()
        for m in models:
            require(isinstance(m, dict), 'model must be an object')
            mid = m['model_id']
            require(isinstance(mid, str) and bool(mid.strip()) and mid not in seen, 'invalid or duplicate model ID')
            seen.add(mid)
            require(isinstance(m['provider'], str) and bool(m['provider'].strip()), 'invalid provider')
            require(type(m['callable']) is bool and tokens(m['context_window_tokens']), 'invalid model')
            require(type(m['supports_tools']) is bool, 'supports_tools must be bool')
            modalities = m['input_modalities']
            require(isinstance(modalities, list) and all(isinstance(x, str) and bool(x) for x in modalities), 'invalid modalities')
            require(len(modalities) == len(set(modalities)), 'duplicate modalities')
            require(isinstance(m['pricing'], dict) and all(number(m['pricing'][k]) for k in ('input_usd_per_1m', 'output_usd_per_1m')), 'invalid prices')
            scores = m['quality_scores']
            require(isinstance(scores, dict) and set(scores) <= DIMENSIONS and all(v is None or number(v) for v in scores.values()), 'invalid quality scores')
            require(m['quality_source'] is None or isinstance(m['quality_source'], str), 'invalid quality source')
            require(isinstance(m['price_date'], str), 'invalid price date')
    except ContractError:
        raise
    except (KeyError, TypeError, AttributeError, ValueError, OverflowError) as exc:
        raise ContractError(f'Missing catalog field or invalid JSON shape ({type(exc).__name__})') from exc


def _validate_inputs(request, plan, candidates, predictions, models):
    validate_request(request)
    for obj in (plan, candidates, predictions):
        envelope(obj, request['request_id'])
    tasks = plan['subtasks']
    require(isinstance(tasks, list) and len(tasks) <= 4, 'invalid subtasks')
    ids = [t['subtask_id'] for t in tasks]
    require(all(isinstance(s, str) and s.startswith('T') and s[1:].isdigit() for s in ids), 'invalid subtask ID')
    require(plan['direct_task']['subtask_id'] == 'DIRECT', 'DIRECT missing')
    for task in tasks + [plan['direct_task']]:
        require(isinstance(task['instruction'], str) and bool(task['instruction']), 'missing instruction')
        require(isinstance(task['dimensions'], list) and bool(task['dimensions']) and set(task['dimensions']) <= DIMENSIONS, 'invalid dimensions')
        require(task['output_type'] == 'text', 'V1 runner supports text outputs only')
        confidence = task.get('dimension_confidence', {})
        require(isinstance(confidence, dict) and set(confidence) <= set(task['dimensions']) and all(number(x) and x <= 1 for x in confidence.values()), 'invalid confidence')
        deps = task['depends_on']
        require(isinstance(deps, list) and len(deps) == len(set(deps)) and set(deps) <= set(ids) and task['subtask_id'] not in deps, 'invalid dependencies', 'INVALID_DAG')
        refs = task['input_refs']
        require(isinstance(refs, list) and all(isinstance(ref, str) for ref in refs), 'invalid input_refs', 'INVALID_DAG')
        require(len(refs) == len(set(refs)) and all(ref == 'original_prompt' or (ref.endswith('.output') and ref[:-7] in deps) for ref in refs), 'invalid input_refs', 'INVALID_DAG')
    require(not plan['direct_task']['depends_on'], 'DIRECT cannot have dependencies', 'INVALID_DAG')
    topological(tasks)
    validate_models(models)
    model_map = {m['model_id']: m for m in models}
    eligible = candidates['eligible_models']
    require(isinstance(eligible, dict) and set(eligible) == set(ids) | {'DIRECT'}, 'candidate task coverage mismatch')
    for mids in eligible.values():
        require(isinstance(mids, list) and len(mids) == len(set(mids)) and set(mids) <= set(model_map), 'invalid candidate model IDs')
    expected = {(s, m) for s, mids in eligible.items() for m in mids}
    rows = predictions['predictions']
    require(isinstance(rows, list), 'predictions must be a list')
    keys = [(p['subtask_id'], p['model_id']) for p in rows]
    require(len(keys) == len(set(keys)) and set(keys) == expected, 'prediction coverage mismatch')
    require(isinstance(predictions['predictor_version'], str) and bool(predictions['predictor_version']), 'missing predictor_version')
    require(all(tokens(p[k]) for p in rows for k in ('input_tokens_est', 'output_tokens_est')), 'invalid predicted tokens')


def validate_inputs(request, plan, candidates, predictions, models):
    try:
        _validate_inputs(request, plan, candidates, predictions, models)
    except ContractError:
        raise
    except (KeyError, TypeError, AttributeError, ValueError, OverflowError) as exc:
        raise ContractError(f'Missing field or invalid JSON shape ({type(exc).__name__})') from exc


def _validate_call(result, request_id, subtask_id, model_id):
    envelope(result, request_id)
    require(result['subtask_id'] == subtask_id and result['model_id'] == model_id, 'LLC response identity mismatch')
    require(result['status'] in ('succeeded', 'failed'), 'invalid call status')
    require(result['usage_source'] in ('provider', 'estimated', 'unavailable'), 'invalid usage source')
    require(number(result['latency_ms']), 'invalid latency')
    for k in ('input_tokens', 'output_tokens'):
        require(result[k] is None or tokens(result[k]), 'invalid usage')
    if result['usage_source'] != 'unavailable':
        require(tokens(result['input_tokens']) and tokens(result['output_tokens']), 'missing usage')
    else:
        require(result['input_tokens'] is None and result['output_tokens'] is None, 'unavailable usage must use null tokens')
    if result['status'] == 'succeeded':
        require(isinstance(result['text'], str) and result['error'] is None, 'invalid successful call')
    else:
        require(result['text'] is None or isinstance(result['text'], str), 'invalid failed call text')
        require(isinstance(result['error'], dict) and all(isinstance(result['error'].get(k), str) and bool(result['error'][k]) for k in ('code', 'message')), 'missing call error')


def validate_call(result, request_id, subtask_id, model_id):
    try:
        _validate_call(result, request_id, subtask_id, model_id)
    except ContractError:
        raise
    except (KeyError, TypeError, AttributeError, ValueError, OverflowError) as exc:
        raise ContractError(f'Missing call field or invalid JSON shape ({type(exc).__name__})') from exc
