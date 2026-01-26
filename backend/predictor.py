from sqlalchemy.orm import Session
import models
import json
import logging
import re
import random
import time
import requests

logger = logging.getLogger(__name__)

class Predictor:
    def __init__(self, db: Session):
        self.db = db
        
    def get_settings(self):
        settings = self.db.query(models.AppSettings).all()
        return {s.key: s.value for s in settings}

    def predict(self, lottery_type):
        result = self.predict_multi(lottery_type, count=1)
        if "error" in result:
            return result
        first = (result.get("predictions") or [{}])[0]
        return {"analysis": result.get("analysis", ""), "prediction": first}

    def predict_multi(self, lottery_type: str, count: int = 5):
        cfg = self._lottery_config(lottery_type)
        if not cfg:
            return {"error": "Invalid lottery type"}

        history = (
            self.db.query(models.LotteryRecord)
            .filter_by(lottery_type=lottery_type)
            .order_by(models.LotteryRecord.issue.desc())
            .limit(100)
            .all()
        )
        if not history:
            return {"error": "No data available. Please scrape data first."}

        settings = self.get_settings()
        api_key = settings.get("llm_api_key") or ""
        base_url = settings.get("llm_base_url") or ""
        model = settings.get("llm_model", "gpt-3.5-turbo")

        if not api_key or not model:
            return {"error": "LLM not configured"}

        llm = self._predict_with_llm(lottery_type, cfg, history, api_key, base_url, model, count)
        if llm and "error" not in llm:
            validated = self._validate_predictions(cfg, llm.get("predictions", []), count)
            if validated:
                return {
                    "analysis": llm.get("analysis", ""),
                    "predictions": validated,
                    "meta": llm.get("meta") or {"used_llm": True, "model": model, "base_url": base_url},
                }
        return {"error": "LLM failed"}

    def _lottery_config(self, lottery_type: str):
        if lottery_type == "ssq":
            return {"red_max": 33, "red_pick": 6, "blue_max": 16, "blue_pick": 1}
        if lottery_type == "dlt":
            return {"red_max": 35, "red_pick": 5, "blue_max": 12, "blue_pick": 2}
        return None

    def _pad2(self, n: int) -> str:
        return f"{n:02d}"

    def _parse_nums(self, s: str) -> list[int]:
        if not s:
            return []
        out = []
        for x in str(s).split(","):
            x = x.strip()
            if not x:
                continue
            if x.isdigit():
                out.append(int(x))
        return out

    def _recent_evaluation_summary(self, lottery_type: str, limit: int = 60) -> str:
        rows = (
            self.db.query(models.PredictionRecord)
            .filter_by(lottery_type=lottery_type, evaluated=True)
            .order_by(models.PredictionRecord.created_at.desc())
            .limit(limit)
            .all()
        )
        if not rows:
            return ""
        items = []
        for r in rows:
            items.append(
                {
                    "based_on_issue": r.based_on_issue,
                    "actual_issue": r.actual_issue,
                    "red_hits": r.red_hits,
                    "blue_hits": r.blue_hits,
                    "total_hits": r.total_hits,
                    "pred_red": r.red_balls,
                    "pred_blue": r.blue_balls,
                    "act_red": r.actual_red_balls,
                    "act_blue": r.actual_blue_balls,
                }
            )
        return json.dumps(items, ensure_ascii=False)

    def _predict_with_llm(self, lottery_type: str, cfg: dict, history, api_key: str, base_url: str, model: str, count: int):
        data_lines = []
        for h in reversed(history[:30]):
            data_lines.append(f"Issue: {h.issue}, Red: {h.red_balls}, Blue: {h.blue_balls}")
        data_str = "\n".join(data_lines)

        prompt = f"""
你是中国彩票({lottery_type.upper()})号码分析助手。请基于最近开奖数据生成下一期候选号码组合。

要求：
1) 只返回 JSON（不要 markdown，不要多余文字）。
2) JSON 结构必须为：{{"predictions":[...]}}，不要输出 analysis 字段。
3) "predictions" 必须恰好包含 {count} 组，且每组互不重复。
4) 每组必须包含：
   - "red_balls": {cfg["red_pick"]} 个不重复号码字符串，范围 "01"~"{cfg["red_max"]:02d}"，升序
   - "blue_balls": {cfg["blue_pick"]} 个不重复号码字符串，范围 "01"~"{cfg["blue_max"]:02d}"，升序

最近 30 期数据（新→旧顺序已省略，按行表示）：
{data_str}
""".strip()

        t0 = time.monotonic()
        try:
            max_tokens = min(1600, 200 + (count * 80))
            base = (base_url or "https://api.siliconflow.cn/v1").rstrip("/")
            url = f"{base}/chat/completions"
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.2,
            }
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            logger.info(f"LLM Call Start: model={model}, count={count}")
            last_exc = None
            for attempt in range(3):
                try:
                    r = requests.post(url, headers=headers, json=payload, timeout=(15, 300))
                    logger.info(f"LLM Call Success: attempt={attempt + 1}, status={r.status_code}")
                    break
                except (requests.Timeout, requests.ConnectionError) as e:
                    last_exc = e
                    logger.warning(f"LLM Call Failed: attempt={attempt + 1}, error={str(e)}")
                    if attempt < 2:
                        time.sleep(2)
                    continue
            else:
                logger.error(f"LLM Call Failed: all retries exhausted")
                raise last_exc or requests.Timeout("LLM request timeout")
            if r.status_code < 200 or r.status_code >= 300:
                return {"error": f"LLM Call Failed: HTTP {r.status_code}", "raw_response": (r.text or "")[:500]}
            data = r.json() if r.content else {}
            content = (
                ((data.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
            ).strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]

            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                match = re.search(r"\{.*\}", content, re.DOTALL)
                if match:
                    parsed = json.loads(match.group(0))
                return {"error": "Failed to parse LLM response as JSON", "raw_response": content}
            latency_ms = int((time.monotonic() - t0) * 1000)
            meta = {
                "used_llm": True,
                "model": model,
                "base_url": base_url,
                "latency_ms": latency_ms,
            }
            if isinstance(parsed, dict):
                parsed["meta"] = parsed.get("meta") or meta
                return parsed
            return {"error": "Invalid LLM response", "raw_response": content}
        except Exception as e:
            logger.error(f"LLM Error: {e}")
            return {"error": f"LLM Call Failed: {str(e)}"}

    def _validate_predictions(self, cfg: dict, preds: list, count: int):
        seen = set()
        out = []
        for item in preds or []:
            reds = item.get("red_balls") or []
            blues = item.get("blue_balls") or []
            if not isinstance(reds, list) or not isinstance(blues, list):
                continue
            reds_i = []
            blues_i = []
            ok = True
            for x in reds:
                if not isinstance(x, str) or not x.isdigit():
                    ok = False
                    break
                v = int(x)
                if v < 1 or v > cfg["red_max"]:
                    ok = False
                    break
                reds_i.append(v)
            for x in blues:
                if not isinstance(x, str) or not x.isdigit():
                    ok = False
                    break
                v = int(x)
                if v < 1 or v > cfg["blue_max"]:
                    ok = False
                    break
                blues_i.append(v)
            if not ok:
                continue
            if len(set(reds_i)) != cfg["red_pick"] or len(reds_i) != cfg["red_pick"]:
                continue
            if len(set(blues_i)) != cfg["blue_pick"] or len(blues_i) != cfg["blue_pick"]:
                continue
            reds_s = [self._pad2(v) for v in sorted(reds_i)]
            blues_s = [self._pad2(v) for v in sorted(blues_i)]
            key = (",".join(reds_s), ",".join(blues_s))
            if key in seen:
                continue
            seen.add(key)
            out.append({"red_balls": reds_s, "blue_balls": blues_s})
            if len(out) >= count:
                break
        return out

    def _weights_from_history(self, cfg: dict, history):
        red_counts = {i: 0 for i in range(1, cfg["red_max"] + 1)}
        blue_counts = {i: 0 for i in range(1, cfg["blue_max"] + 1)}
        for h in history:
            for n in self._parse_nums(h.red_balls):
                if 1 <= n <= cfg["red_max"]:
                    red_counts[n] += 1
            for n in self._parse_nums(h.blue_balls):
                if 1 <= n <= cfg["blue_max"]:
                    blue_counts[n] += 1
        return red_counts, blue_counts

    def _weights_from_evaluations(self, cfg: dict, lottery_type: str, limit: int = 200):
        red_boost = {i: 0.0 for i in range(1, cfg["red_max"] + 1)}
        blue_boost = {i: 0.0 for i in range(1, cfg["blue_max"] + 1)}
        rows = (
            self.db.query(models.PredictionRecord)
            .filter_by(lottery_type=lottery_type, evaluated=True)
            .order_by(models.PredictionRecord.created_at.desc())
            .limit(limit)
            .all()
        )
        for r in rows:
            if r.total_hits is None:
                continue
            act_red = set(self._parse_nums(r.actual_red_balls))
            act_blue = set(self._parse_nums(r.actual_blue_balls))
            for n in self._parse_nums(r.red_balls):
                if n in act_red:
                    red_boost[n] += 1.0 + (r.total_hits / 10.0)
            for n in self._parse_nums(r.blue_balls):
                if n in act_blue:
                    blue_boost[n] += 1.0 + (r.total_hits / 10.0)
        return red_boost, blue_boost

    def _sample_without_replacement(self, items: list[int], weights: dict[int, float], k: int):
        pool = list(items)
        picked = []
        for _ in range(min(k, len(pool))):
            total = 0.0
            for n in pool:
                total += max(0.0, float(weights.get(n, 0.0)))
            if total <= 0:
                choice = random.choice(pool)
            else:
                r = random.random() * total
                acc = 0.0
                choice = pool[-1]
                for n in pool:
                    acc += max(0.0, float(weights.get(n, 0.0)))
                    if acc >= r:
                        choice = n
                        break
            pool.remove(choice)
            picked.append(choice)
        return picked

    def _predict_with_heuristics(self, lottery_type: str, cfg: dict, history, count: int):
        red_hist, blue_hist = self._weights_from_history(cfg, history)
        red_boost, blue_boost = self._weights_from_evaluations(cfg, lottery_type)

        red_w = {i: 1.0 + red_hist[i] * 0.35 + red_boost[i] * 0.15 for i in red_hist}
        blue_w = {i: 1.0 + blue_hist[i] * 0.45 + blue_boost[i] * 0.20 for i in blue_hist}

        seen = set()
        out = []
        tries = 0
        while len(out) < count and tries < count * 50:
            tries += 1
            reds = self._sample_without_replacement(list(range(1, cfg["red_max"] + 1)), red_w, cfg["red_pick"])
            blues = self._sample_without_replacement(list(range(1, cfg["blue_max"] + 1)), blue_w, cfg["blue_pick"])
            reds_s = [self._pad2(v) for v in sorted(reds)]
            blues_s = [self._pad2(v) for v in sorted(blues)]
            key = (",".join(reds_s), ",".join(blues_s))
            if key in seen:
                continue
            seen.add(key)
            out.append({"red_balls": reds_s, "blue_balls": blues_s})
        return out
