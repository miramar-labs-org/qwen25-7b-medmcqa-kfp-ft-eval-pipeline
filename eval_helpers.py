import re as _re


def extract_answer(text):
    """Normalize model output to a comparable answer token."""
    t = text.strip().lower()
    # MCQ letter (a-d)
    m = _re.search(r'\b([a-d])\b', t)
    if m:
        return m.group(1).upper()
    # fallback: first token if it's a single letter
    first = t.split()[0] if t else ""
    if len(first) == 1 and first.isalpha():
        return first.upper()
    return first


def _make_user_content(row):
    return row["instruction"]


def make_infer_fn(tokenizer, model, system_message, max_new_tokens, do_sample):
    import torch

    def _infer(row):
        messages = [{"role": "user", "content": _make_user_content(row)}]
        ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)
        with torch.no_grad():
            out = model.generate(
                ids, max_new_tokens=max_new_tokens, do_sample=do_sample,
                pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(out[0][ids.shape[-1]:], skip_special_tokens=True).strip()

    return _infer
