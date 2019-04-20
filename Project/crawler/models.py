import requests
import re
from urllib.parse import urljoin

from bs4 import BeautifulSoup

USER_AGENT = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.139 Safari/537.36'


class AmazonCrawler:
    def __init__(self):
        self.base_url = 'https://www.amazon.com/s'

    def query(self, product):
        query_string = {
            'ref': 'nb_sb_noss_1',
            'k': '{}'.format(product)
        }
        headers = {
            'user-agent': USER_AGENT,
        }
        response = requests.get(self.base_url, headers=headers, params=query_string)
        html = response.text
        return self.parse(html)

    def parse(self, html):
        soup = BeautifulSoup(html, 'html.parser')
        product_list = list()
        for div in soup.find_all('div', class_='s-result-item'):
            product_aTag = div.find('h5').find('a', class_='a-link-normal')
            if product_aTag is None:
                continue
            product_imgSrc = div.find('img').get('src')

            product_name = product_aTag.find('span').get_text()
            product_link = urljoin('http://www.amazon.com', product_aTag.get('href'))

            product_price_whole = div.find('span', class_='a-price-whole')
            if product_price_whole is None:
                continue
            product_price_fractional = div.find('span', class_='a-price-fraction')
            if product_price_fractional is None:
                continue

            product_price = float(product_price_whole.get_text()+product_price_fractional.get_text())
            product_list.append((product_imgSrc, product_name, product_link, product_price, 'Amazon'))
        return product_list


class EbayCrawler:
    def __init__(self):
        self.base_url = 'https://www.ebay.com/sch/i.html'

    def query(self, product):
        query_string = {
            '_nkw': '{}'.format(product)
        }
        headers = {
            'user-agent': USER_AGENT
        }
        response = requests.get(self.base_url, headers=headers, params=query_string)
        html = response.text

        return self.parse(html)

    def parse(self, html):
        soup = BeautifulSoup(html, 'html.parser')
        product_list = list()
        for li in soup.find_all('li', class_='s-item'):
            product_img = li.find('img', class_='s-item__image-img')
            product_imgSrc = product_img.get('src')
            if product_imgSrc.endswith('.gif'):
                product_imgSrc = product_img.get('data-src')
            product_aTag = li.find('a', class_='s-item__link')
            product_link = product_aTag.get('href')

            product_name = product_aTag.find('h3').string
            if product_name is None:
                continue
            price_str = li.find('span', class_='s-item__price').string
            if price_str is None:
                continue
            product_price = float(price_str[1:].replace(',', ''))
            product_list.append((product_imgSrc, product_name, product_link, product_price, 'eBay'))
        return product_list



if __name__ == '__main__':
    c = AmazonCrawler()
    c.query('chair')
    # print(c.query('chair'))