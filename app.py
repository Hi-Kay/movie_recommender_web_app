from flask import Flask
from recommender import recommend_random
from recommender import recommend_neighborhood
from recommender import recommend_with_NMF
from flask import render_template
from flask import request
from utils import movies 
from utils import example_query
from utils import movie_to_id


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', name='Heike', movies = movies.title.to_list())
    
@app.route('/recommender')
def recommendations():
    print(request.args)
        
    titles = request.args.getlist('title')
    print(titles)

    movies_id = []
    for title in titles:
        movie_id = int(movie_to_id(title))
        movies_id.append(movie_id)
    print( movies_id)
    
    ratings = request.args.getlist('Ratings')
    user_input = dict(zip(movies_id,ratings))

    for keys in user_input:
        user_input[keys] = int(user_input[keys])
       
    print(user_input)

    if request.args['algo']=='Random':
        recs = recommend_random()
        return render_template('recommendations.html',recs =recs)
    
    if request.args['algo']=='Cosine similarity':
        recs = recommend_neighborhood(user_input, k = 3)
        return render_template('recommendations.html',recs =recs)
    if request.args['algo']=='NMF':
        recs = recommend_with_NMF(user_input, k = 3)
        return render_template('recommendations.html',recs =recs)
    else:
        return f"Function not defined"  
    
if __name__ == "__main__":
    app.run(debug=True, port=5000)
