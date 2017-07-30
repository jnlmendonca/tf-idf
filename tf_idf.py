"""TF-IDF

Perform Term Frequency - Inverse Document Frequency over a set of
documents.

See README for instructions.
"""

import math
import operator
import os
import pickle

from collections import Counter


class TfIdf():

    def __init__(self, compute_on_add=True):
        """Constructor

        Arguments:
            compute_on_add: boolean with the desired state for the variable
                with the same name. If True, the results will be computed
                whenever a document is added. Otherwise, they will be
                computed when results are exported ore used to score a
                document. Defaults to True.

        Returns:
            An instance of TfIdf class.
        """
        self._dataset = {}
        self._tf_idf_results = {}
        self._results_up_to_date = False
        self._compute_on_add = compute_on_add

    def set_compute_on_add(self, state):
        """Change when results are computed

        Set the value of the 'compute on add' variable. If True,
        the results will be computed whenever a document is added. Otherwise,
        they will be computed when results are exported ore used to
        score a document.

        Arguments:
            state: boolean

        Returns:
            None
        """

        # Validate state
        if type(state) is not bool:
            raise TypeError("state must be a boolean")

        # Set state
        self._compute_on_add = state

    def add_document(self, doc_name, doc_terms):
        """Add document

        Add a new document with its corresponding set of terms to the dataset.

        Arguments:
            doc_name: a string with the document's name
            doc_terms: a list of terms

        Returns:
            None
        """

        # Count distinct terms in document
        doc_term_counter = Counter(doc_terms)
        doc_term_dict = dict(doc_term_counter)

        # Get document from summary
        doc = self._dataset.get(doc_name, self._new_doc())

        # Add new document terms to document
        doc["term_count"] = {
            t: doc["term_count"].get(t, 0) + doc_term_dict.get(t, 0)
            for t in set(doc["term_count"]) | set(doc_term_dict)
        }
        doc["nr_terms"] += len(doc_terms)

        # Renew document in summary
        self._dataset[doc_name] = doc

        # Flag that dataset changed
        self._results_up_to_date = True

        # Compute results
        if self._compute_on_add:
            self._compute_results()

    def _compute_results(self):
        """Compute results

        Uses the current dataset to compute the TF-IDF results.
        This method should not be run manually and is instead run automatically
        according to the 'compute on add' state set. This defaults to True and
        may be changed using the 'setself._compute_on_add()' method.
        """

        # Evaluate if results have to be computed
        if not self._results_up_to_date:
            return

        # Compute term frequency
        term_frequencies = self._tf()

        # Compute inverse document frequency
        inverse_document_frequencies = self._idf()

        # Compute tf_idf results
        for term in inverse_document_frequencies:
            for document in term_frequencies:
                if term in term_frequencies[document]:
                    term_tf_idf = term_frequencies[document][term] * \
                        inverse_document_frequencies[term]

                    if term not in self._tf_idf_results:
                        self._tf_idf_results[term] = {}

                    self._tf_idf_results[term][document] = term_tf_idf

        # Flag that the results were computed
        self._results_up_to_date = True

    def score_document(self, doc_terms):
        """Score document

        Use the computed results to score a list of terms against all
        the documents added to the dataset. A score is provided for each
        document in the dataset. The biggest score corresponds to the
        document most similar to the provided list of terms.

        Arguments:
            doc_terms: a list of terms

        Returns:
            An ordered list of tuples containing document names and their
            TF-IDF score. The tuples are ordered by score form highest to
            lowest.
        """

        if not self._compute_on_add:
            self._compute_results()

        # Count distinct terms in document
        doc_term_counter = Counter(doc_terms)
        doc_term_dict = dict(doc_term_counter)

        doc_sums = {}
        for term in doc_term_dict:
            if term in self._tf_idf_results:
                for document in self._tf_idf_results[term]:
                    if document not in doc_sums:
                        doc_sums[document] = 0

                    doc_sums[document] += doc_term_dict[term] * \
                        self._score_term(term, document)

        # Sort documents by score
        sorted_doc_sums = sorted(
            doc_sums.items(),
            key=operator.itemgetter(1),
            reverse=True
        )

        return sorted_doc_sums

    def export(self, file_path, include_dataset=True):
        """Export data

        Export data to an external file so it can be saved and loaded later.

        Arguments:
            file_path: the path to the exported file
            includeself._dataset: True if the dataset should be exported along
            with the results. False otherwise. Defaults to True.

        Returns:
            True if the data was successfully exported. False otherwise.
        """

        # Validate file path
        if not file_path:
            return False

        # Create results directory if it does not exist
        if not os.path.exists(os.path.dirname(file_path)):
            try:
                os.makedirs(os.path.dirname(file_path))
            except:
                return False

        # Compute results
        if not self._compute_on_add:
            self._compute_results()

        # Prepare payload
        payload = {}
        payload["results"] = self._tf_idf_results
        if include_dataset:
            payload["dataset"] = self._dataset

        try:
            # Export data
            with open(file_path, "wb") as file:
                pickle.dump(payload, file)
                return True
        except:
            return False

    def load(self, file_path):
        """ELoad data

        Load data from an external file so it can be incremented and/or used to
        score documents.

        Arguments:
            file_path: the path to the exported file

        Returns:
            True if the data was successfully loaded. False otherwise.
        """

        # Validate file path
        if not file_path:
            return False

        try:
            # Open file and retrieve data
            with open(file_path, "rb") as file:
                payload = pickle.load(file)
                self._tf_idf_results = payload.get("results", {})
                self._dataset = payload.get("dataset", {})
                return True
        except:
            return False

    def _new_doc(self):
        """Add new document to the dataset

        Initialize a new document in the dataset. This method should not be
        run manually.
        """

        new_doc = {}
        new_doc["nr_terms"] = 0
        new_doc["term_count"] = {}
        return new_doc

    def _tf(self):
        """Term Frequency

        Compute the term frequency over the dataset. This method should not be
        run manually.
        """

        # Initialize results
        term_frequencies = {}

        # Compute term frequencies per document
        for document in self._dataset:
            frequencies = {
                t: self._dataset[document]["term_count"][t] /
                self._dataset[document]["nr_terms"]
                for t in self._dataset[document]["term_count"]
            }
            term_frequencies[document] = frequencies

        return term_frequencies

    def _idf(self):
        """Inverse Document Frequency

        Compute the inverse document frequency over the dataset. This method
        should not be run manually.
        """

        # Initialize results
        inverse_document_frequencies = {}

        # Compute total number of documents
        nr_documents = len(self._dataset)

        # Count documents where term appears
        for document in self._dataset:
            document_appearances = dict.fromkeys(
                self._dataset[document]["term_count"],
                1
            )
            inverse_document_frequencies = {
                t: inverse_document_frequencies.get(t, 0) +
                document_appearances.get(t, 0)
                for t in set(inverse_document_frequencies) |
                set(document_appearances)
            }

        inverse_document_frequencies = {
            t: -1 * math.log(inverse_document_frequencies[t] / nr_documents)
            for t in inverse_document_frequencies
        }

        return inverse_document_frequencies

    def _score_term(self, term, document):
        """Term score function"""

        return self._tf_idf_results[term][document]
