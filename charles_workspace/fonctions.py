import concurrent.futures
from transformers import pipeline

def modiliserSujetDesArticles(titre_list, batch_size=10):
    classifier = pipeline(model="facebook/bart-large-mnli")
    result_list = []

    candidate_labels=[
    "Environnement",
    "Politique",
    "COVID",
    "Technologie",
    "Économie",
    "Justice",
    "Éducation",
    "Santé",
    "Sport"
    ]


    # Process texts in batches
    with concurrent.futures.ThreadPoolExecutor() as executor:

        for i in range(0, len(titre_list), batch_size):
            batch = titre_list[i:i + batch_size]
            results = executor.map(classifier, batch, [candidate_labels]*len(batch))
            for res in results:
                try: 
                    final_result = max(zip(res['scores'], res['labels']))[1]
                    result_list.append(final_result)
                except Exception as e:
                    print('KeyError:', e)
                    print('result :', res )
                    pass

    return result_list