from huggingface_hub import HfApi
api = HfApi()
api.create_repo("empbetty/tangyuan-dataset-background-removed", repo_type="dataset")
api.upload_folder(
    folder_path="/Users/empbetty/Documents/SCPD/cs236/project/tangyuan_dataset",
    repo_id="empbetty/tangyuan-dataset-2",
    repo_type="dataset",
)
