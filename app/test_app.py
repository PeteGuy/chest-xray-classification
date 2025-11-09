import pytest
from fastapi.testclient import TestClient
from .main import app

client = TestClient(app)
test_file ="app/test_person1_virus_7.jpeg"
test_file_bad ="app/test_bad_file.txt"
def test_hello():
    response = client.get("/")
    assert response.status_code==200
    assert response.json() == {"message":"You are currently on the chest xray classification API !"}

def test_metadata():
    response = client.get("/metadata")
    assert response.status_code == 200
    assert response.json() == {"Model":"ResNet-18",
            "Training specifications":"40 epochs, finetuning over image-net backbone, binary sigmoid classification over chest Xray images",
            "Metrics used":"Precision, recall"
            }
    response = client.get("/metadata?get_summary=True")
    assert response.status_code == 200
    jsonr = response.json()
    assert ("Torchinfo summary" in jsonr and jsonr["Torchinfo summary"] != None )

@pytest.mark.parametrize("file_path,show_attention,status_code,json_response_keys",[
    (test_file,False,200,["Score","Class"]),
    (test_file, True, 200, ["Score", "Class","attention_url"]),
    (test_file_bad, False, 422, ["detail"])
])
def test_predict(file_path,show_attention,status_code,json_response_keys):
    '''
    :param file_path: path of the file, can be either our test image or a bad file expected to fail
    :param show_attention:
    :param status_code: expected status code
    :param json_response_keys: expected result keys
    '''
    with open(file_path,"rb") as f:
        body = {"file":f}
        response = client.post("/predict",files=body,params={"show_attention":show_attention})
    assert response.status_code == status_code
    assert list(response.json().keys()) == json_response_keys

    if show_attention:
        # Make sure we can get the attention image and that it gets deleted right after
        img_url = response.json()["attention_url"]
        response = client.get(url=img_url)
        assert response.status_code == 200
        response = client.get(url=img_url)
        assert response.status_code == 404


@pytest.mark.parametrize("files_paths,status_code,keys,sub_keys,num_good_files",[
    ([test_file,test_file,test_file],200,["results"],["name","score","class"],3),
    ([test_file, test_file_bad, test_file], 200, ["message","results"], ["name", "score", "class"],2),
    ([test_file_bad,test_file_bad,test_file_bad], 422, ["detail"], [], 0),

])
def test_predict_batch(files_paths,status_code,keys,sub_keys,num_good_files):
    '''
    :param files_paths: 3 files to be used
    :param status_code: expected status code
    :param keys: expected keys
    :param sub_keys: expected subkeys
    :param num_good_files: number of results we expect, if one or more of the file is bad (not jpeg or png) we should not return a score for it in our results
    '''
    with open(files_paths[0],"rb") as f1:
        with open(files_paths[1], "rb") as f2:
            with open(files_paths[2], "rb") as f3:
                response = client.post(url="/predict_batch",files=[
                    ('files',f1),
                    ('files',f2),
                    ('files',f3),
                ])
    assert response.status_code == status_code
    json_rep = response.json()
    assert list(json_rep.keys()) == keys
    if "results" in keys:
        assert len(json_rep["results"]) == num_good_files
        assert list(json_rep["results"][0].keys()) == sub_keys


