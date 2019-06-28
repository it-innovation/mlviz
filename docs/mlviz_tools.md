# The feature importance tool

This tool provides filter methods for gauging the importance of different features, prior to fitting any model. It can help guide which features to visualise in different MLViz tools.

# The HD visualisation tool

The focus of the High-dimensional (HD) visualisation tool is on visualising a representation of a high dimensional dataset in two dimensions to aid the exploratory analysis process. This tool allows the users to manually extract clusters of data points to feed into other visualisation tools to, for example, evaluate class seperability in high dimensional space. 

Tips when using the tool:

+ Take note of how the different dimensionality reduction techniques scale (see appendix of the user guide), e.g., using t-SNE for large data sets (n>10000) will take a long time! 
+ The parameters of the dimensionality algorthim can be passed as kwargs to HDViz if the default parameters are not sufficient.

#  The Draughtsman plot tool

The draughtsman plot (a.k.a. a pairplot) is an interactive tool for inspecting data points in the original feature space.

Tips when using the tool:

+ While the plot can plot up to 10 features, for performance reasons we recommended plotting no more than 5 features at once.