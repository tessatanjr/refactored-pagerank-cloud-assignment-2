# refactored-pagerank-cloud-assignment-2
pagerank.py contains the code for section 2 & 3, Numerical comparison between iterative page rank and closed form page rank.

crawl_ranker.py implements the adapted page rank algorithm for **(ii) AI webcrawling for training or otherwise**
OpenAI uses specialized web crawlers like GPTBot (https://developers.openai.com/api/docs/bots/) to collect training data for foundation models with website owners able to control access via Web Robot configuration (https://www.robotstxt.org/). Imagine you are designing a crawling strategy for a new AI research organization and must decide how to prioritize which newly-discovered web pages to crawl first using a GPTBot-like crawler. Write a program/software that takes a small directed web graph (represented as a dictionary of URLs and their outlinks) and precomputed PageRank scores, then returns the top k URLs to crawl based on authority, explaining why high-PageRank pages might yield higher-quality training data for generative AI models. Propose one heuristic to find high-quality pages that permit crawling.


To run this:
- create and activate venv
- Run `pip install -r requirements.txt`
- add open ai api key to crawl_ranker.py
- install __web-Google_10k.txt__ provided by Google in 2002 available at: __https://hunglvosu.github.io/posts/2020/07/PA3/__ and place it in project directory
- run `python3 pagerank.py`
- run `python3 crawl_ranker.py`
