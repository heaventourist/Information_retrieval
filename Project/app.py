from flask import Flask
from flask import request
from flask import render_template
import json

from crawler.models import AmazonCrawler, EbayCrawler

app = Flask(__name__, static_url_path='/static')

amazon_crawler = AmazonCrawler()
ebay_crawler = EbayCrawler()

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/query', methods=['GET'])
def search():
    product_list = []
    if 'product' in request.args:
        product = request.args.get('product').strip()
        if product:
            product_list = amazon_crawler.query(product)
            product_list.extend(ebay_crawler.query(product))
            product_list = sorted(product_list, key=lambda p: p[-2])
    return json.dumps(product_list)


if __name__ == '__main__':
    app.run(debug=False)

