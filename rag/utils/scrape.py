"""
# ml-playground/rag/utils/scrape
#
# Copyright (C) 2024 Wojciech Polak
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
"""

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from .logging_config import configure_logging
from .env import env

logger = configure_logging('scraper')


def get_links(url: str, domain: str, depth: int, visited=None) -> list[str]:
    # Check if the URL has already been visited or reached the depth limit
    if visited is None:
        visited = set()
    if url in visited or depth == 0:
        return []

    if env.WL_BEARER_TOKEN:
        headers = {'Authorization': f'Bearer {env.WL_BEARER_TOKEN}'}
    elif env.WL_COOKIE_TOKEN:
        headers = {'Cookie': env.WL_COOKIE_TOKEN}
    else:
        headers = {}

    # Fetch HTML content
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logger.error(f'Error fetching {url}: {e}')
        return []

    # Parse HTML content
    soup = BeautifulSoup(response.text, 'html.parser')

    # Extract links
    links = []
    for link in soup.find_all('a', href=True):
        href = link['href']

        # Convert relative links to absolute links
        absolute_link = urljoin(url, href)

        # Check if the link is within the specified domain
        if urlparse(absolute_link).netloc == domain:
            links.append(absolute_link)

    # Mark URL as visited
    visited.add(url)
    logger.debug('GET %s', url)

    # Recursively fetch links from the extracted links with reduced depth
    for link in links:
        links.extend(get_links(link, domain, depth - 1, visited))

    return links
