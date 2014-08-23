## Practical machine learning project
===================

### Project folder for practical machine learning


This project analyses a data set collected from measurement devices attached on 6 participants
while they were doing weight lifting exercises, then proceeds to build a series of random forest
models to predict the manner in which they did the exercise. A k-fold (k=10) cross validation
is performed on the subdivisions of the training data; the missclassification rates and OOB
error rates are calculated for each fold step; a final model is selected based these rates;
and the final model is used to precdict 20 different test cases.

### Thanks

The data files for this project come from this source: http://groupware.les.inf.puc-rio.br/har.
Thanks for the generousity of the authors.
