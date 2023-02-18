# Recomendation-Systems

Although it emerged in the 1990s, it gains more importance in the 2009s.
Recommendation systems do not recommend products or services to users using certain techniques.
Our overall goal: We will try to filter the products, with abundant content, to make smart suggestions to the user based on their own choices. So our goal is to reduce them and make better suggestions.

Simple Recommender System: General recommendations made with business knowledge or simple techniques. Top scorers of the category, trending ones, legends etc.

Association Rule Learning: Product recommendations are made according to the rules learned through association analysis.It extracts the probabilities of the products that are bought together very often and provides the opportunity to make specific recommendations based on them.It is a rule-based machine learning technique used to find patterns (pattern, relationship, structure) in data.

Content Based Filtering: These are systems that make recommendations based on the meta information, descriptions and features of the products. 
They are distance-based methods that make recommendations based on product similarity. 
They are systems that make a recommendation according to the description and category information of the product.

So how are we going to translate the contents here close to each other? We will perform these operations by converting them to certain metrics.

1) Vectorization of Texts: It is the process of converting texts into mathematical expressions and bringing them into a measurable form. 
It can be accomplished using two different methods. These;

  a) Count Vector: Euclidean distance of each line is found by finding the number of words in here.
  
  b) TF-IDF: It performs a normalization operation over the crossover frequencies of the words both in their own text and in the whole focused data. 
  It fixes any bias in the Count vector.
 
Collaborative Filtering: These are the methods that reflect the common opinions of the community on the basis of user or product. If a community liked one movie and liked the other, the following assumption can be made; If people like movie a and they like movie b, movie b can be recommended to anyone who likes movie a but doesn't like movie b.

Apriori Algorithm (Apriori Algorithm): It is a basket analysis method. It is used to reveal product associations. There are 3 basic metrics that we will calculate on this algorithm. With these, we have the opportunity to observe the relationship patterns and structures in the data set. We have the opportunity to evaluate these relations with a statistical criterion. These;

1) Support(X,Y) = Freq(X,Y)/N
Probability of X and Y occurring together/all values. (Probability of X and Y occurring together)

2) Confidence(x,Y) = Freq(X,Y) / Freq(X)
Probability of X and Y appearing together/frequency of X. (Probability of selling Y when X is bought)

It is an interconnected metric.

3) Lift = Support(X,Y) / (Support(X)*Support(Y)).
(When X is purchased, the probability of Y being purchased increases by the lift value.)

How Does Apriori Work?

It is the elimination process according to the threshold support and frequency value of a particular product.

The sieving process is done by looking at the freq and sup values of each product. 

Then the product pairs are examined. And a new list is created by always filtering and eliminating. From this list, support confidence and lift values are calculated and observed.

# Business Problem

Armut, Turkey's largest online service platform, brings together service providers and those who want to receive service. 
It provides easy access to services such as cleaning, modification and transportation with a few touches on your computer or smart phone.
It is desired to create a product recommendation system with Association Rule Learning by using the data set containing the service users and the services and categories these users have received.


Data set

The data set consists of the services customers receive and the categories of these services.
It contains the date and time information of each service received.

UserId: Customer number
ServiceId: Anonymized services belonging to each category. (Example: Upholstery washing service under the cleaning category)

A ServiceId can be found under different categories and refers to different services under different categories.

(Example: The service with CategoryId 7 and ServiceId 4 is honeycomb cleaning, while the service with CategoryId 2 and ServiceId 4 is furniture assembly)

CategoryId: Anonymized categories. (Example: Cleaning, transportation, renovation category)

CreateDate: The date the service was purchased



