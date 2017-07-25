# TF-IDF
## Term Frequency - Inverse Document Frequency

This is a very simple TF-IDF implementation, aimed at easy and quick experimentation. It is not aimed at efficiency, so don't rely on it for important projects!

### Basic usage
1. Instantiate the TfIdf class
	```python
	from tf_idf import TfIdf

	tfidf = TfIdf()
	```
2. Add documents
	```python
	tfidf.add_document("Article1", ["i", "love", "cats"])
	tfidf.add_document("Article2", ["i", "love", "cake"])
	tfidf.add_document("Article3", ["cats", "love", "cake"])
	```

3. Score another document
	```python
	tfidf.score_document(["you", "love", "cats", "and", "cake"])
	```

	which results in
	```python
	[('Article3', 0.2703100720721096), ('Article1', 0.1351550360360548), ('Article2', 0.1351550360360548)]
	```

4. Export data
	```python
	tfidf.export("path/to/file.pkl")
	```

5. Load data
	```python
	tfidf.load("path/to/file.pkl")
	```


### Special cases
+ 	Want to add a lot of documents and use results only at the end?

  	After instantiating the TfIdf class, set the 'compute on add variable' to False
	```python
	from tf_idf import TfIdf

	tfidf = TfIdf()
	tfidf.set_compute_on_add(False)
	```

+ 	Want to incrementally build each article?

  	If you want to evaluate an arbitrary number of articles and divide them into a predetermined set of classes, just keep adding the term lists to the same, previously added documents.

  	For example, if you want to divide articles by their main subject: "Cats", "Cake" or "Other"
	```python
	from tf_idf import TfIdf

	tfidf = TfIdf()

	tfidf.add_document("Cats", ["i", "love", "cats"])
	tfidf.add_document("Cake", ["i", "love", "cake"])
	tfidf.add_document("Cats", ["cats", "love", "cake"])
	tfidf.add_document("Cake", ["cats", "love", "cake"])
	tfidf.add_document("Other", ["i", "love", "books"])

	tfidf.score_document(["i", "have", "many", "books"])
	```

	which results in
	```python
	[('Other', 0.3662040962227032), ('Cats', 0.0), ('Cake', 0.0)]
	```


