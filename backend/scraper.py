import requests
from bs4 import BeautifulSoup
from datetime import datetime
from sqlalchemy.orm import Session
import models
import logging

logger = logging.getLogger(__name__)

class LotteryScraper:
    def __init__(self, db: Session):
        self.db = db
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

    def _parse_date(self, date_str: str):
        date_str = (date_str or "").strip()
        if not date_str:
            return None
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d"):
            try:
                return datetime.strptime(date_str, fmt).date()
            except ValueError:
                continue
        return None

    def _fetch_table_rows(self, url: str):
        try:
            response = requests.get(url, headers=self.headers, timeout=20)
            response.raise_for_status()
            response.encoding = 'utf-8'
            soup = BeautifulSoup(response.text, 'html.parser')
            tdata = soup.find('tbody', id='tdata')
            if not tdata:
                return []
                
            return tdata.find_all('tr') or []
        except Exception as e:
            logger.error(f"Error fetching lottery table: {e}")
            return []

    def scrape_ssq(self, limit=100, upsert: bool = False, want_issue: str | None = None):
        want_issue = (want_issue or "").strip() or None
        urls = [
            f"https://datachart.500.com/ssq/history/newinc/history.php?limit={limit}&sort=0",
            f"https://datachart.500.com/ssq/history/newinc/history.php?limit={limit}&sort=1",
        ]
        seen: dict[str, dict] = {}
        for url in urls:
            rows = self._fetch_table_rows(url)
            if not rows:
                continue
            for row in rows:
                cols = row.find_all('td')
                if len(cols) < 16:
                    continue
                issue = cols[0].text.strip()
                if not issue:
                    continue
                reds = ",".join([cols[i].text.strip() for i in range(1, 7)])
                blue = cols[7].text.strip()
                date = self._parse_date(cols[15].text.strip())
                if date is None:
                    continue
                seen[issue] = {"issue": issue, "date": date, "red_balls": reds, "blue_balls": blue}
            if want_issue and want_issue in seen:
                break

        added = 0
        updated = 0
        for issue, payload in seen.items():
            existing = self.db.query(models.LotteryRecord).filter_by(lottery_type='ssq', issue=issue).first()
            if not existing:
                record = models.LotteryRecord(
                    lottery_type='ssq',
                    issue=issue,
                    date=payload["date"],
                    red_balls=payload["red_balls"],
                    blue_balls=payload["blue_balls"],
                )
                self.db.add(record)
                added += 1
                continue
            if not upsert:
                continue
            changed = False
            if payload.get("date") and getattr(existing, "date", None) != payload["date"]:
                existing.date = payload["date"]
                changed = True
            if (getattr(existing, "red_balls", "") or "").strip() != payload["red_balls"]:
                existing.red_balls = payload["red_balls"]
                changed = True
            if (getattr(existing, "blue_balls", "") or "").strip() != payload["blue_balls"]:
                existing.blue_balls = payload["blue_balls"]
                changed = True
            if changed:
                updated += 1

        if added or updated:
            self.db.commit()
        return {"added": added, "updated": updated, "seen": len(seen)}

    def scrape_dlt(self, limit=100, upsert: bool = False, want_issue: str | None = None):
        want_issue = (want_issue or "").strip() or None
        urls = [
            f"https://datachart.500.com/dlt/history/newinc/history.php?limit={limit}&sort=0",
            f"https://datachart.500.com/dlt/history/newinc/history.php?limit={limit}&sort=1",
        ]
        try:
            seen: dict[str, dict] = {}
            for url in urls:
                rows = self._fetch_table_rows(url)
                if not rows:
                    continue
                for row in rows:
                    cols = row.find_all('td')
                    if len(cols) < 15:
                        continue
                    issue = cols[0].text.strip()
                    if not issue:
                        continue
                    reds = ",".join([cols[i].text.strip() for i in range(1, 6)])
                    blues = ",".join([cols[i].text.strip() for i in range(6, 8)])
                    date = self._parse_date(cols[14].text.strip())
                    if date is None:
                        continue
                    seen[issue] = {"issue": issue, "date": date, "red_balls": reds, "blue_balls": blues}
                if want_issue and want_issue in seen:
                    break

            added = 0
            updated = 0
            for issue, payload in seen.items():
                existing = self.db.query(models.LotteryRecord).filter_by(lottery_type='dlt', issue=issue).first()
                if not existing:
                    record = models.LotteryRecord(
                        lottery_type='dlt',
                        issue=issue,
                        date=payload["date"],
                        red_balls=payload["red_balls"],
                        blue_balls=payload["blue_balls"],
                    )
                    self.db.add(record)
                    added += 1
                    continue
                if not upsert:
                    continue
                changed = False
                if payload.get("date") and getattr(existing, "date", None) != payload["date"]:
                    existing.date = payload["date"]
                    changed = True
                if (getattr(existing, "red_balls", "") or "").strip() != payload["red_balls"]:
                    existing.red_balls = payload["red_balls"]
                    changed = True
                if (getattr(existing, "blue_balls", "") or "").strip() != payload["blue_balls"]:
                    existing.blue_balls = payload["blue_balls"]
                    changed = True
                if changed:
                    updated += 1

            if added or updated:
                self.db.commit()
            return {"added": added, "updated": updated, "seen": len(seen)}
        except Exception as e:
            logger.error(f"Error scraping DLT: {e}")
            return {"added": 0, "updated": 0, "seen": 0}
