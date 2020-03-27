# Update 11/14: You are no longer required to handle unknown / missing / "don't know" values.

#     (50%) Implement the k-means clustering algorithm and the HAC (Hierarchical Agglomerative Clustering) algorithm. Link to Skeleton Code
#         HAC
#             HAC should support both single link and complete link options.
#             Since HAC automatically generates groupings for all values of k, you will pass in a range or set of k values for which actual output will be generated.
#         k-means
#             For k-means you will pass in a specific k value for the number of clusters that should be in the resulting clustering.  
#         Distance metric: Use Euclidean distance for continuous attributes a̶n̶d̶ ̶(̶0̶,̶1̶)̶ ̶d̶i̶s̶t̶a̶n̶c̶e̶s̶ ̶f̶o̶r̶ ̶u̶n̶k̶n̶o̶w̶n̶ ̶a̶t̶t̶r̶i̶b̶u̶t̶e̶s̶.̶
#         Only handle continuous features: While we could use mechanisms similar to the KNN lab for handling nominal features, we will assume you are working with just continuous features for this lab.
#         H̶a̶n̶d̶l̶i̶n̶g̶ ̶m̶i̶s̶s̶i̶n̶g̶ ̶v̶a̶l̶u̶e̶s̶:̶ ̶I̶g̶n̶o̶r̶e̶ ̶m̶i̶s̶s̶i̶n̶g̶ ̶v̶a̶l̶u̶e̶s̶ ̶w̶h̶e̶n̶ ̶c̶a̶l̶c̶u̶l̶a̶t̶i̶n̶g̶ ̶c̶e̶n̶t̶r̶o̶i̶d̶s̶ ̶a̶n̶d̶ ̶a̶s̶s̶i̶g̶n̶ ̶t̶h̶e̶m̶ ̶a̶ ̶d̶i̶s̶t̶a̶n̶c̶e̶ ̶o̶f̶ ̶1̶ ̶w̶h̶e̶n̶ ̶d̶e̶t̶e̶r̶m̶i̶n̶i̶n̶g̶ ̶t̶o̶t̶a̶l̶ ̶s̶u̶m̶ ̶s̶q̶u̶a̶r̶e̶d̶ ̶e̶r̶r̶o̶r̶.̶ ̶ ̶ ̶
#             H̶o̶w̶e̶v̶e̶r̶,̶ ̶i̶f̶ ̶a̶l̶l̶ ̶t̶h̶e̶ ̶i̶n̶s̶t̶a̶n̶c̶e̶s̶ ̶i̶n̶ ̶a̶ ̶c̶l̶u̶s̶t̶e̶r̶ ̶h̶a̶v̶e̶ ̶d̶o̶n̶’̶t̶ ̶k̶n̶o̶w̶ ̶f̶o̶r̶ ̶o̶n̶e̶ ̶o̶f̶ ̶t̶h̶e̶ ̶a̶t̶t̶r̶i̶b̶u̶t̶e̶s̶,̶ ̶t̶h̶e̶n̶ ̶u̶s̶e̶ ̶"̶d̶o̶n̶’̶t̶ ̶k̶n̶o̶w̶"̶ ̶i̶n̶ ̶t̶h̶e̶ ̶c̶e̶n̶t̶r̶o̶i̶d̶ ̶f̶o̶r̶ ̶t̶h̶a̶t̶ ̶a̶t̶t̶r̶i̶b̶u̶t̶e̶.̶
#         Handling Distance Ties: when a node or cluster has the same distance to another cluster, which should be rare, just go with the earliest cluster in your list.  
#         The output for the algorithm tested should include for each clustering:
#             the number of clusters,
#             the total SSE of the full clustering. 
#             For each cluster report:
#                 the centroid value
#                 the number of instances tied to that centroid
#                 the SSE of that cluster
#                     The sum squared error (SSE) of a single cluster is the sum of the squared euclidean distance of each cluster member to the cluster centroid.
#             Debug/Evaluation
#                 For debug/evaluation, attributes should be normalized using the formula the formula (x-xmin)/(xmax-xmin)
#                 Use the first k instances in your training data as your initial centroids for k-means
#     (25%) Run both algorithms on the full iris data set where you do not include the output label as one of the input features.
#         For k-means you should always choose k random points in the data set as initial centroids.  
#         If you ever end up with any empty clusters in k-means, re-run with different initial centroids.  
#         Run it for k = 2-7.  
#         State whether you normalize or not (your choice).  
#         Graph the total SSE for each k and discuss your results (i.e. what kind of clusters are being made).
#         Now do it again where you include the output label as one of the input features and discuss your results and any differences.  
#         For this final data set, also run k-means 5 times with k=4, each time with different initial random centroids and discuss any variations in the results.
#     (25%) Run the SK versions of k-means and HAC on iris and compare your results.
#         Find a data set of your choice (i.e. not one previously used for this lab) and use the SK version to get results
#         Experiment with and discuss your results with different hyper-parameters.
#         Compare the utility of some of the clustering quality metrics (separability, compactness (SSE), silhouette, etc.) on different clusterings. For instance, address some of the following:  
#             In your view, which clustering(s) appear to be the best? Justify your answer.
#             How useful is each metric in determining the best clustering?
#             Which metrics support the clustering(s) you chose?
#             Why might these metrics be more relevant given the specific data/task you chose?
#             You may want to use a silhouette score for this problem, in which case you can use sklearn.metrics.silhouette_score. Please state in your report if you coded your own silhouette score function to receive the extra credit points (described below). 
#             Other possible sklearn.metrics you might try (* metrics require ground truth labels):
#                 adjusted_mutual_info_score*
#                 adjusted_rand_score*
#                 homogeneity_score*
#                 completeness_score*
#                 fowlkes_mallows_score*
#                 silhouette_score
#                 calinski_harabasz_score
#                 davies_bouldin_score
#     (optional 5% extra credit) For one of your clustering experiments, calculate (with your own work (code, spreadsheet, etc.) and not a tool you find) and report the average silhouette score for each clustering for k = 2-7.  You do not need to supply full Silhouette graphs, but you could if you wanted to. Discuss how helpful Silhouette appeared to be for selecting which clustering is best.

