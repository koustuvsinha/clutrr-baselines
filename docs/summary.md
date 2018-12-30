## Summary

### Idea

Any NLP problem can broken down into 3 steps:

* Deciding the structure to represent the given sentence/document etc.
* Given the structure, encoding the given sentence/document etc into an latent representation.
* Decoding the latent representation for the downstream task.

Much of the NLP work assumes that the optimal representational structure is the word order given in the sentence and starts applying machine learning models from step 2. Some works attempt to use dependency tree or other linguistic structures instead of the word order. while such linguistic structures are easy to obtain for sentences/documents, their adoption remains limited as they do not provide much advantage over the word order based approach. We hypothesise that learning the structural representation is an important problem in itself.

### Task

For now, we are looking at the task of abstractive summarization.

### Model

We would have following models:

* f<sub>SentenceEncoder</sub>
  * Generate the latent representation corresponding to words in a given sentence.
  * Would most likely be LSTM.
* f<sub>EdgeEncoder</sub>
  * Provide representation for an edge given a pair of nodes (entities).
  * Would most likely be FF.
* f<sub>EdgeTransition</sub>
  * Capture the transition in edge representation across different sentences.
  * Would most likely be a LSTM.
* f<sub>GraphPool</sub>
  * Capture the representation corresponding to the entire graph.
  * Could be something as simple as average operator.

Our contribution would be in terms of learning structure and encoding the given sentences/documents. As such, we are free to use any decoder.
