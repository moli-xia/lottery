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

    def scrape_ssq(self, limit=100):
        url = f"http://datachart.500.com/ssq/history/newinc/history.php?limit={limit}&sort=0"
        try:
            response = requests.get(url, headers=self.headers)
            response.encoding = 'utf-8'
            soup = BeautifulSoup(response.text, 'html.parser')
            tdata = soup.find('tbody', id='tdata')
            if not tdata:
                logger.error("Could not find tbody id='tdata' in SSQ response")
                return 0
                
            rows = tdata.find_all('tr')
            
            count = 0
            for row in rows:
                cols = row.find_all('td')
                if len(cols) < 16: # Ensure enough columns
                    continue
                
                try:
                    issue = cols[0].text.strip()
                    reds = ",".join([cols[i].text.strip() for i in range(1, 7)])
                    blue = cols[7].text.strip()
                    # sales = cols[13].text.strip().replace(',', '')
                    # pool = cols[14].text.strip().replace(',', '')
                    date_str = cols[15].text.strip()
                    try:
                        date = datetime.strptime(date_str, '%Y-%m-%d').date()
                    except ValueError:
                         # Fallback if date format is weird or missing
                        continue

                    existing = self.db.query(models.LotteryRecord).filter_by(lottery_type='ssq', issue=issue).first()
                    if not existing:
                        record = models.LotteryRecord(
                            lottery_type='ssq',
                            issue=issue,
                            date=date,
                            red_balls=reds,
                            blue_balls=blue,
                        )
                        self.db.add(record)
                        count += 1
                except Exception as row_e:
                    logger.warning(f"Error parsing SSQ row: {row_e}")
                    continue
            
            self.db.commit()
            return count
        except Exception as e:
            logger.error(f"Error scraping SSQ: {e}")
            return 0

    def scrape_dlt(self, limit=100):
        url = f"http://datachart.500.com/dlt/history/newinc/history.php?limit={limit}&sort=0"
        try:
            response = requests.get(url, headers=self.headers)
            response.encoding = 'utf-8'
            soup = BeautifulSoup(response.text, 'html.parser')
            tdata = soup.find('tbody', id='tdata')
            if not tdata:
                logger.error("Could not find tbody id='tdata' in DLT response")
                return 0
                
            rows = tdata.find_all('tr')
            
            count = 0
            for row in rows:
                cols = row.find_all('td')
                if len(cols) < 15:
                    continue
                
                try:
                    issue = cols[0].text.strip()
                    reds = ",".join([cols[i].text.strip() for i in range(1, 6)])
                    blues = ",".join([cols[i].text.strip() for i in range(6, 8)])
                    date_str = cols[14].text.strip()
                    try:
                        date = datetime.strptime(date_str, '%Y-%m-%d').date()
                    except ValueError:
                        continue

                    existing = self.db.query(models.LotteryRecord).filter_by(lottery_type='dlt', issue=issue).first()
                    if not existing:
                        record = models.LotteryRecord(
                            lottery_type='dlt',
                            issue=issue,
                            date=date,
                            red_balls=reds,
                            blue_balls=blues
                        )
                        self.db.add(record)
                        count += 1
                except Exception as row_e:
                    logger.warning(f"Error parsing DLT row: {row_e}")
                    continue
            
            self.db.commit()
            return count
        except Exception as e:
            logger.error(f"Error scraping DLT: {e}")
            return 0
