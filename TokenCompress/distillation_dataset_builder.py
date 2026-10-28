"""
TeacherDatasetBuilder — teacher 모델이 생성한 (query, response) 쌍으로
KD 학습용 데이터셋을 만드는 객체.
"""
import torch
from tqdm.auto import tqdm


class TeacherDatasetBuilder:
    def __init__(self, model, tokenizer, builder):
        """
        builder: PromptBuilder 인스턴스. 이미 만들어둔 trainer.builder를 그대로
                 넘기거나, 별도로 생성한 PromptBuilder를 넘겨도 됨.
                 여기서 쓰는 건 build_kd()의 teacher_prefix 조립 로직뿐이므로,
                 target_token_ids/ae_token_id는 이 클래스 입장에선 의미 없음
                 (student 쪽은 안 씀 — teacher 프롬프트만 필요).
        """
        self.model = model
        self.tokenizer = tokenizer
        self.builder = builder
        self.im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    def _build_teacher_prefix(self, query):
        """PromptBuilder의 조립 로직을 그대로 재사용 — teacher 쪽 prefix만 필요."""
        built = self.builder.build_kd(query)
        return built["teacher_prefix"], built["user_content"]

    @torch.no_grad()
    def generate_one(self, query, max_new_tokens=256, gold_label=None):
        """단일 쿼리에 대해 teacher 응답 하나를 생성."""
        teacher_prefix, user_content = self._build_teacher_prefix(query)

        inputs = self.tokenizer(teacher_prefix, add_special_tokens=False,
                                return_tensors="pt").to(self.model.device)
        prompt_len = inputs["input_ids"].shape[1]

        out = self.model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            eos_token_id=[self.tokenizer.eos_token_id, self.im_end_id],
            pad_token_id=self.tokenizer.eos_token_id,
        )
        resp_ids = out[0, prompt_len:].cpu().tolist()
        resp_text = self.tokenizer.decode(resp_ids, skip_special_tokens=True).strip()

        record = {
            "query": query,
            "response_ids": resp_ids,
            "response": resp_text,
        }
        if gold_label is not None:
            gold_label = str(gold_label).strip()
            record["gold_label"] = gold_label
            record["matches_gold"] = (resp_text == gold_label)
        return record

    def build(self, dataset, query_key="text", label_key=None, max_new_tokens=256, verbose=True):
        """
        dataset: 각 원소가 dict-like이고 query_key(필요시 label_key)를 갖고 있는 시퀀스.
        label_key: None이면 gold_label 비교 없이 생성만 함(태스크에 고정 정답이 없을 때,
                   예: summarize/persona처럼 정답이 하나로 고정되지 않는 태스크).
        """
        data = []
        n = len(dataset)
        iterator = tqdm(range(n), desc="teacher 응답 생성") if verbose else range(n)

        for i in iterator:
            sample = dataset[i]
            query = sample[query_key]
            gold_label = sample[label_key] if label_key is not None else None
            data.append(self.generate_one(query, max_new_tokens=max_new_tokens, gold_label=gold_label))

        if verbose and label_key is not None:
            n_match = sum(d["matches_gold"] for d in data)
            print(f"teacher가 gold와 일치: {n_match}/{len(data)} ({100*n_match/len(data):.1f}%)")

        return data