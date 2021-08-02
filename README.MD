# OpenCV for face recognition
Used libraries:

 **Pandas**
 
  **Numpy** 
  
  **Scipy** 
  
  **Sklearn**
  
  **Implicit**
  
  **Seaborn**
  
  ## Idea of recommendation system
  
  Mine recommendation system has three steps 
  
1. At first I will find similar users with nearest neighbours algorithm and create a dict. Key will be user and the 
value is the list of the most rated (viewed) videos witch were liked by users similar to 'key' user.
  
2. Secondly, I will find similar movies by genre.

3. At third stage I will find 10 similar movies to each movie that I recommended for users.
  
 If the user is new then I recommend top 10 videos
  ## Installation and running

1. Clone the repo
```
$ git clone https://gitlab.com/digital-hq/rnd/recomendation.git
```

2. Create a Python virtual environment named 'test' and activate it
```
$ virtualenv test
```
```
$ source test/bin/activate
```

3. Run the following command in your Terminal/Command Prompt to install the libraries required
```
$ pip3 install -r requirements.txt
```

4. To recognize face masks in real-time video streams type the following command:

```
$ python3 main.py
```

## Functions

File 'main.py' contains six functions for working with recommendation system

update_similar_video() - will update similarity matrix for videos

get_similar_book(book_id) - will find similar book by the given one

update_similar_user_book() - will update similarity matrix for users

get_similar_user_book() - will return recommendations for all users found by the same user

get_top_10_popular() - will return top ten popular videos 

update_all() - will update similarity matrix for videos and for users