import arxiv
import os
import re

import arxiv
import os
import re

def extract_research_papers(path: str, subject: str = "Astronomy OR AI", num: int = 5):
    os.makedirs(path, exist_ok=True)

    def clean_filename(title):
        # Remove invalid characters: \ / : * ? " < > |
        return re.sub(r'[\\/*?:"<>|]', "_", title) + ".pdf"

    search = arxiv.Search(
        query=subject,
        max_results=num,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )

    for result in search.results():
        try:
            filename = clean_filename(result.title)
            # Downloads PDF to dirpath
            pdf_path = result.download_pdf(dirpath=path)
            # Rename the file to clean filename
            final_path = os.path.join(path, filename)
            os.rename(pdf_path, final_path)
            print(f"Downloaded: {filename}")
        except Exception as e:
            print(f"Failed to download {result.title}: {e}")