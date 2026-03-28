import math
import json
from openai import OpenAI
from collections import deque

client = OpenAI(api_key="insert open ai key here")

web_graph = {
    "https://www.science.org/doi/abs/10.1126/science.1160379": ["https://dl.acm.org/doi/epdf/10.1145/1052934.1052938", "https://hunglvosu.github.io/posts/2020/07/PA3/"],
    "https://dl.acm.org/doi/epdf/10.1145/1052934.1052938": ["https://tetr.io/", "https://www.tomscott.com/usvsth3m/maths/"],
    "https://snap.stanford.edu/class/cs224w-readings/Brin98Anatomy.pdf": ["https://tetr.io/", "https://claude.ai/new"],
    "https://hunglvosu.github.io/posts/2020/07/PA3/": ["https://tetr.io/", "https://www.mcdonalds.com.sg/", "https://www.tomscott.com/usvsth3m/maths/"],
    "https://www.mcdonalds.com.sg/": ["https://www.science.org/doi/abs/10.1126/science.1160379", "https://claude.ai/new"],
    "https://github.com": ["https://tetr.io/", "https://openai.com", "https://www.tomscott.com/usvsth3m/maths/"],
    "https://openai.com": ["https://www.tomscott.com/usvsth3m/maths/"],
    "https://www.tomscott.com/usvsth3m/maths/": ["https://claude.ai/new"],
    "https://claude.ai/new": ["https://tetr.io/", "https://www.tomscott.com/usvsth3m/maths/"],
    "https://tetr.io/": ["https://claude.ai/new", "https://www.tomscott.com/usvsth3m/maths/", "https://snap.stanford.edu/class/cs224w-readings/Brin98Anatomy.pdf"],
    "https://arxiv.org": ["https://dl.acm.org/doi/epdf/10.1145/1052934.1052938", "https://www.science.org/doi/abs/10.1126/science.1160379", "https://snap.stanford.edu/class/cs224w-readings/Brin98Anatomy.pdf"],
    "https://dl.acm.org": ["https://www.science.org/doi/abs/10.1126/science.1160379","https://dl.acm.org/doi/epdf/10.1145/1052934.1052938", "https://snap.stanford.edu/class/cs224w-readings/Brin98Anatomy.pdf"],
    "https://www.cognizant.com/sg/en": ["https://claude.ai/new", "https://www.mcdonalds.com.sg/"],
    "https://www.foxnews.com/": ["https://www.cognizant.com/sg/en", "https://arxiv.org"],
}

robots_txt = {
    "https://www.science.org/doi/abs/10.1126/science.1160379": True,
    "https://dl.acm.org/doi/epdf/10.1145/1052934.1052938": True,
    "https://snap.stanford.edu/class/cs224w-readings/Brin98Anatomy.pdf": True,
    "https://hunglvosu.github.io/posts/2020/07/PA3/": True,
    "https://www.mcdonalds.com.sg/": False, 
    "https://github.com": True,
    "https://openai.com": False, 
    "https://www.tomscott.com/usvsth3m/maths/": True,
    "https://claude.ai/new": False,
    "https://tetr.io/": False,
    "https://arxiv.org": True,
    "https://dl.acm.org": True,
    "https://www.cognizant.com/sg/en": True, 
    "https://www.foxnews.com/": True,
}

trusted_seeds = [
    "https://arxiv.org",
    "https://dl.acm.org",
]

domains = {
    ".edu": "academic",
    ".gov": "government",
    ".ac.": "academic",
    ".org": "organisation",
    ".com": "company",
    ".io":  "startup",
    ".co":  "startup",
    ".net": "other",
}

def pagerank_url(web_graph, damping=0.85, iterations=100):
    nodes = list(web_graph.keys())
    N = len(nodes)

    ranks = {}
    for node in nodes:
        ranks[node] = 1 / N

    dangling_nodes = []
    for n in nodes:
        if len(web_graph[n]) == 0:
            dangling_nodes.append(n)

    for _ in range(iterations):
        dangling_sum = 0
        for n in dangling_nodes:
            dangling_sum += ranks[n]

        updated = {}
        for node in nodes:
            updated[node] = (1 - damping) / N
            updated[node] += damping * dangling_sum / N

        for node in nodes:
            out_links = web_graph[node]
            if len(out_links) == 0:
                continue
            share = ranks[node] / len(out_links)
            for to in out_links:
                if to in updated:
                    updated[to] += damping * share

        ranks = updated

    return ranks

def compute_hop_distances(web_graph, seeds):
    distances = {}
    visited = set()
    queue = deque()

    for seed in seeds:
        if seed in web_graph:
            queue.append((seed, 0))
            visited.add(seed)
            distances[seed] = 0

    while queue:
        node, dist = queue.popleft()
        for neighbour in web_graph.get(node, []):
            if neighbour not in visited:
                visited.add(neighbour)
                distances[neighbour] = dist + 1
                queue.append((neighbour, dist + 1))

    # Nodes not reachable from seeds get max penalty
    for node in web_graph:
        if node not in distances:
            distances[node] = 999

    return distances

def get_domain_type(url):
    for tld, domain_type in domains.items():
        if tld in url:
            return domain_type
    return "other"

def llm_quality_score(url, domain_type):
    safe_url = url.encode('utf-8').decode('ascii', errors='ignore')  
    if domain_type == "academic":
        prompt = f"""
You are determining the authority and quality of webpages to rank them based on quality for data extraction.
You do not need to scrape or visit the URL. Based on the domain rely on your knowledge of the company.
This is the URL for you to evaluate: {safe_url}

The webpage contains academic content, evaluate it using the listed criterias below.
Each criteria is assigned a certain number of points. Add up the total score attained by the page based on the various criteria.
If you are unsure of whether the webpage fits a certain criteria, assign a score of 0.
The maximum score is 10. If a score above 10 is obtained, return 10.

List of criteria:
- Published by a recognised academic institution or journal: +3
- Content is peer reviewed: +2
- Publisher is associated or works closely with a reputable academic institution: +2
- Frequently cited in academic literature: +2
- Founded more than 10 years ago: +1
- Open access: +1
Total possible: 10

Return only a single number between 1-10 as the final score, with no additional explanation.
"""

    elif domain_type == "organisation":
        prompt = f"""
You are determining the authority and quality of webpages to rank them based on quality for data extraction.
You do not need to scrape or visit the URL. Based on the domain rely on your knowledge of the company.
This is the URL for you to evaluate: {safe_url}

The webpage seems to be from an ORGANISATION, evaluate it using the listed criterias below.
Each criteria is assigned a certain number of points. Add up the total score attained by the page based on the various criteria.
If you are unsure of whether the webpage fits a certain criteria, assign a score of 0.
The maximum score is 10. If a score above 10 is obtained, return 10.

Scoring criteria:
- Publishing organisation is internationally recognised (e.g. WHO, IEEE, ACM, UNESCO): +3
- Organisation is known to produce highly reliable content : +2
- Organisation is associated with many reputable organisations: +1
- Referenced by government or academic sources: +1
- Founded more than 10 years ago: +1
- Known to be politically unbiased: +1
Total possible: 10

Return only a single number between 1-10 as the final score, with no additional explanation.
"""

    elif domain_type == "government":
        prompt = f"""
You are determining the authority and quality of webpages to rank them based on quality for data extraction.
You do not need to scrape or visit the URL. Based on the domain rely on your knowledge of the company.
This is the URL for you to evaluate: {safe_url}

The webpage seems to be a GOVERNMENT source, evaluate it using the listed criterias below.
Each criteria is assigned a certain number of points. Add up the total score attained by the page based on the various criteria.
If you are unsure of whether the webpage fits a certain criteria, assign a score of 0.
The maximum score is 10. If a score above 10 is obtained, return 10.

Scoring criteria:
- Government body is a scientific or statistical agency: +3
- URL is independently verified by reliable third parties: +2
- Government body frequently publishes transparent methodologies: +2
- Referenced or cited by academic institutions or independent researchers: +1
- Website has been active and maintained for more than 10 years: +1
- Country has no known state censorship and has high press freedom: +1
Total possible: 10 

Return only a single number between 1-10 as the final score, with no additional explanation.
"""

    elif domain_type == "company":
        prompt = f"""
You are determining the authority and quality of webpages to rank them based on quality for data extraction.
You do not need to scrape or visit the URL. Based on the domain rely on your knowledge of the company.
This is the URL for you to evaluate: {safe_url}

The webpage seems to be from a COMPANY, evaluate it using the listed criterias below.
Each criteria is assigned a certain number of points. Add up the total score attained by the page based on the various criteria.
If you are unsure of whether the webpage fits a certain criteria, assign a score of 0.
The maximum score is 10. If a score above 10 is obtained, return 10.

Scoring criteria:
- Company is well-known and established +3
- Comapny is founded 10 or more years ago: +2
- Company publishes highly reliable research: +2
- Publicly listed company: +1
- Has associated peer-reviewed research: +1
- Referenced by academic or government sources: +1
Total possible: 10

Return only a single number between 1-10 as the final score, with no additional explanation.
"""

    else: 
        prompt = f"""
You are determining the authority and quality of webpages to rank them based on quality for data extraction.
You do not need to scrape or visit the URL. Based on the domain rely on your knowledge of the company.
This is the URL for you to evaluate: {safe_url}

Evaluate it using the listed criterias below.
Each criteria is assigned a certain number of points. Add up the total score attained by the page based on the various criteria.
If you are unsure of whether the webpage fits a certain criteria, assign a score of 0.
The maximum score is 10. If a score above 10 is obtained, return 10.

Scoring criteria:
- Has a known reputation for quality content: +3
- Referenced by other credible sources: +2
- Content is original: +2
- Has been active for more than 4 years: +1
- Primary purpose is for the publishing of academic research: +2
Total possible: 10 

Return only a single number between 1-10 as the final score, with no additional explanation.
"""
        
    response = client.chat.completions.create(
        model="gpt-5",
        messages=[{"role": "user", "content": prompt}]
    )

    raw = response.choices[0].message.content.strip()
    if not raw:
        raise ValueError("LLM returned empty response")
    score = float(raw)
    return score

def combined_score(pagerank, hop_distance, llm_score,
                   pagerank_weight=0.2, pagehop_weight=0.4, llmeval_weight=0.4):

    pagerank_normalized = min(pagerank * 1000, 10)
    hop_score = 10 / (hop_distance + 1)

    final_score = (
        pagerank_weight * pagerank_normalized
        + pagehop_weight * hop_score
        + llmeval_weight * llm_score
    )

    return round(final_score, 4)

if __name__ == "__main__":

    ranks = pagerank_url(web_graph, damping=0.85, iterations=100)
    print("page ranks")
    for url, score in sorted(ranks.items(), key=lambda x: x[1], reverse=True):
        print(f" {score:.6f}  {url}")
    print("\n")

    hop_distances = compute_hop_distances(web_graph, trusted_seeds)
    print("no of hops from trusted source")
    for url, dist in sorted(hop_distances.items(), key=lambda x: x[1]):
        d = str(dist) if dist != 999 else "not reachable"
        print(f"  {d} hops  {url}")
    print("\n")

    allowed_urls = [url for url in web_graph if robots_txt.get(url, False)]
    llm_results = {}

    for url in allowed_urls:
        domain_type = get_domain_type(url)

        try:
            score = llm_quality_score(url, domain_type)
            llm_results[url] = {
                "domain_type": domain_type,
                "llm_score": score
            }
            print(f"  LLM Score: {score}/10  {url}")
        except Exception as e:
            print(f"Error: {e}")
            llm_results[url] = {
                "domain_type": domain_type,
                "llm_score": 5.0
            }
    print("\n")

    final_scores = {}
    for url, data in llm_results.items():
        score = combined_score(
            pagerank=ranks.get(url, 0),
            hop_distance=hop_distances.get(url, 999),
            llm_score=data["llm_score"]
        )
        final_scores[url] = score

    sorted_final = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)

    print("Combined Scores")
    print(f"  {'Rank':<5} {'Final':>7}  {'PR':>8}  {'Hop':>5}  {'LLM':>5}  URL")
    print("  " + "-" * 80)

    for i, (url, score) in enumerate(sorted_final, start=1):
        data = llm_results[url]
        pr = ranks.get(url, 0)
        hop = hop_distances.get(url, 999)
        hop_display = str(hop) if hop != 999 else "999"
        print(f"  {i:<5} {score:>7.4f}  {pr:>8.6f}  {hop_display:>5}  {data['llm_score']:>5}  {url}")

    for i, (url, score) in enumerate(sorted_final[:5], start=1):
        data = llm_results[url]
        print(f"\n  {i}. {url}")
        print(f"     Final Score:   {score:.4f}/10")
        print(f"     Domain Type:   {data['domain_type']}")
        print(f"     LLM Score:     {data['llm_score']}/10")