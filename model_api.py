import uvicorn
from fastapi import FastAPI
import numpy as np
import pickle 
import pandas as pd
from user_input import UserInput
import json
import pandas as pd
import numpy as np
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


df1=pd.read_csv(r'C:\Users\user\Desktop\project implementation\juloh.csv')

app= FastAPI()
pickle_in=open(r"C:\Users\user\Desktop\project implementation\recommend.pkl","rb")
recommender=pickle.load(pickle_in)
list_of_all_imp=df1['imp'].tolist()

@app.get('/test/')
def testjuloh():
    return {"Hello": "World"}

@app.get('/recommend/{user_input}')
async def recommend(user_input:str):
        try:
            find_close_match=difflib.get_close_matches(user_input,list_of_all_imp)
            close_match=find_close_match[0]
            vec= TfidfVectorizer()
            vecs=vec.fit_transform(df1['imp'])
            sim=cosine_similarity(vecs)
            user_id=df1[df1.imp==close_match]["ids"].values[0]
            scores=list(enumerate(sim[user_id]))
            sorted_scores=sorted(scores, key=lambda x:x[1], reverse=True)
            sorted_scores = sorted_scores[1:]
            programs=[df1[programs[0]==df1["ids"]]["Program"].values[0] for programs in sorted_scores]
        
        
            first_ten=[]
            count=0
            for program in programs:
                if count>10:
                    break
                count+=1
                first_ten.append(program)


            df2 = pd.DataFrame(first_ten, columns = ['Programmes'])    
            a=df2['Programmes'].unique()
            df4 = pd.DataFrame(a, columns = ['Programmes'])  
            dict = df4.to_dict()
            df3 = pd.DataFrame (dict, columns = ['Programmes'])
            return df3
        except:
            myexception={'exception':'No recommendations'}
            return myexception


if __name__=='__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
