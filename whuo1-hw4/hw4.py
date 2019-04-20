import logging
import re
import sys
from bs4 import BeautifulSoup
from queue import Queue, PriorityQueue
from urllib import parse, request

logging.basicConfig(level=logging.DEBUG, filename='output.log', filemode='w')
visitlog = logging.getLogger('visited')
extractlog = logging.getLogger('extracted')


def parse_links(root, html):
    soup = BeautifulSoup(html, 'html.parser')
    for link in soup.find_all('a'):
        href = link.get('href')
        if href:
            text = link.string
            if not text:
                text = ''
            text = re.sub('\s+', ' ', text).strip()
            yield (parse.urljoin(root, link.get('href')), text)

def get_relevance(link):
    if 'cs415' in link:
        return 0
    if 'yarowsky' in link:
        return 1
    elif 'cs.jhu.edu' in link:
        return 2
    else:
        return 3

def parse_links_sorted(root, html):
    # TODO: implement
    soup = BeautifulSoup(html, 'html.parser')
    for link in soup.find_all('a'):
        href = link.get('href')
        if href:
            text = link.string
            if not text:
                text = ''
            text = re.sub('\s+', ' ', text).strip()
            url_joined = parse.urljoin(root, link.get('href'))
            yield ((get_relevance(url_joined), url_joined), text)


def get_links(url):
    res = request.urlopen(url)
    return list(parse_links(url, res.read()))

def is_not_self_referencing(from_url, to_url):
    return parse.urlparse(to_url).path != parse.urlparse(from_url).path

def is_non_local(from_url, to_url):
    return parse.urlparse(to_url).netloc != parse.urlparse(from_url).netloc

def get_nonlocal_links(url):
    '''Get a list of links on the page specificed by the url,
    but only keep non-local links and non self-references.
    Return a list of (link, title) pairs, just like get_links()'''

    # TODO: implement
    links = get_links(url)
    filtered = list(filter(lambda link: is_non_local(url, link[0]) and
     is_not_self_referencing(url, link[0]), links))
    return filtered


def crawl(root, wanted_content=[], within_domain=True):
    '''Crawl the url specified by `root`.
    `wanted_content` is a list of content types to crawl
    `within_domain` specifies whether the crawler should limit itself to the domain of `root`
    '''
    # TODO: implement

    queue = Queue()
    queue.put(root)

    visited = set()
    extracted = []
    while not queue.empty():
        # if len(visited) > 50:
        #     break
        url = queue.get()
        # print(url)
        try:
            req = request.urlopen(url)
            if wanted_content and not any([x.lower() in req.headers['Content-Type'].lower() for x in wanted_content]):
                continue
            html = req.read()

            visited.add(url)
            visitlog.debug(url)

            for ex in extract_information(url, html):
                extracted.append(ex)
                extractlog.debug(ex)

            for link, title in parse_links(url, html):
                if link in visited or (not is_not_self_referencing(root, link)) or (within_domain and is_non_local(root, link)):
                    continue
                queue.put(link)

        except Exception as e:
            print(e, url)

    return visited, extracted

def crawl_priority(root, wanted_content=[], within_domain=True):
    '''Crawl the url specified by `root`.
    `wanted_content` is a list of content types to crawl
    `within_domain` specifies whether the crawler should limit itself to the domain of `root`
    '''
    # TODO: implement

    queue = PriorityQueue()
    queue.put((get_relevance(root), root))

    visited = set()
    extracted = []

    while not queue.empty():
        if len(visited) > 50:
            break
        rank, url = queue.get()
        # print(url)
        try:
            req = request.urlopen(url)
            if wanted_content and not any([x.lower() in req.headers['Content-Type'].lower() for x in wanted_content]):
                continue
            html = req.read()

            visited.add(url)
            visitlog.debug(url)

            for ex in extract_information(url, html):
                extracted.append(ex)
                extractlog.debug(ex)

            for (rank, link), title in parse_links_sorted(url, html):
                if link in visited or (not is_not_self_referencing(root, link)) or (within_domain and is_non_local(root, link)):
                    continue
                queue.put((rank, link))

        except Exception as e:
            print(e, url)

    return visited, extracted

def extract_information(address, html):
    '''Extract contact information from html, returning a list of (url, category, content) pairs,
    where category is one of PHONE, ADDRESS, EMAIL'''

    # TODO: implement
    results = []
    for match in re.findall('\d\d\d-\d\d\d-\d\d\d\d', str(html)):
        results.append((address, 'PHONE', match))
    for match in re.findall(r"([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)", str(html)):
        results.append((address, 'EMAIL', match))
    for match in re.findall(r'([A-Za-z]+(\s[A-Za-z]+){0,1}(\s[A-Za-z]+){0,1},\s[A-Za-z]+\.{0,1}\s\d{5})', str(html)):
        results.append((address, 'ADDRESS', match))
    return results


def writelines(filename, data):
    with open(filename, 'w') as fout:
        for d in data:
            print(d, file=fout)


def main():
    site = sys.argv[1]

    links = get_links(site)
    writelines('links.txt', links)

    nonlocal_links = get_nonlocal_links(site)
    writelines('nonlocal.txt', nonlocal_links)

    visited, extracted = crawl_priority(site, ['text'])
    writelines('visited.txt', visited)
    writelines('extracted.txt', extracted)


if __name__ == '__main__':
    main()