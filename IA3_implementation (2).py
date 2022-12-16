
# AI 534
# IA3 implementation code

import pandas as pd, numpy as np, matplotlib.pyplot as plt, sklearn, scipy.sparse, seaborn as sns

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.svm import LinearSVC, SVC

#######################################
# preprocessing
#######################################

# preprocessing data
def preprocessing_data(data, countvectorized:bool=False, sentiment:bool=True, testing:bool=False, val=None):
#If sentiment is true, it checks for positive sentiment
   #Find the index for the positive and negative data 
    def find_index(data):
        positive = []
        negative = []
        for i in range(1, len(data)):
            if data["sentiment"][i]==1:
                positive.append(i)
            else:
                negative.append(i)
        return positive, negative

    #Find the top 10 words
    def find_max_word(indexes, vector_of_words, name_of_words):
        num_words = vector_of_words.shape[1]
        array_word = np.zeros(num_words)
        for index in indexes:
            array_word += vector_of_words[index]
        common_word_index = sorted(range(len(array_word)), key=lambda i: array_word[i], reverse=True)[:10]
        word = name_of_words[common_word_index]
        return word
    
    positive, negative = find_index(data)
    
    if sentiment:
        check = positive
    else:
        check = negative
    
    if countvectorized:
        vectorizer = CountVectorizer(lowercase=True)
        bow = vectorizer.fit_transform(data["text"])
        vectors = bow.toarray()
        words = vectorizer.get_feature_names_out()
        top_words = find_max_word(check, vectors, words)
    else:
        vectorizer = TfidfVectorizer(use_idf=True, lowercase=True)
        bow = vectorizer.fit_transform(data["text"])
        vectors = bow.toarray()
        words = vectorizer.get_feature_names_out()
        top_words = find_max_word(check, vectors, words)
    
    if testing:
        bow = vectorizer.transform(val_data["text"])
        vectors = bow.toarray()
        return vectors
               
    return top_words, vectors


#######################################
# Validation functions
#######################################

#extracts feature matrix and target vectors from data
def blanket(data, val_data):
    y_train = data["sentiment"] 
    y_val = val_data["sentiment"]
    tfid_vectors = scipy.sparse.csr_matrix(preprocessing_data(data)[1])
    val_vectors = scipy.sparse.csr_matrix(preprocessing_data(data, testing=True, val = val_data))
    return y_train, y_val,tfid_vectors, val_vectors

#trains a linear kernel svm per cs value using training data (data) and tests on val_data
def linear_validate(data, val_data, cs):
    y_train, y_val,train_vectors, val_vectors = blanket(data, val_data)
    support = np.zeros(len(cs))
    scores = np.zeros(len(cs))
    for (i, c) in enumerate(cs):
        clf = LinearSVC(C=c)
        clf.fit(train_vectors, y_train)
        decision_function = clf.decision_function(train_vectors)
        support_vector_indices = np.where(np.abs(decision_function) <= 1 + 1e-15)[0]
        scores[i] = clf.score(val_vectors, y_val)
        support[i] = len(support_vector_indices)
    return scores, support

#trains polynomial kernel SVM with degree 2 and tests per cs value on val_data
def quad_validate(data, val_data, cs):
    y_train, y_val,train_vectors, val_vectors = blanket(data, val_data)
    scores = np.zeros(len(cs))
    support = np.zeros(len(cs))
    for (i, c) in enumerate(cs):
        clf = SVC(C=c, kernel='poly', degree=2)
        clf.fit(train_vectors, y_train)
        scores[i] = clf.score(val_vectors, y_val)
        support[i] = sum(clf.n_support_)
    return scores, support

#trains RBF kernel SVM with degree 2 and tests a grid of cs and γs values on val_data
def rbf_validate(data, val_data, cs, γs, fixed_c:bool=False, fixed_γ:bool=False):
    y_train, y_val,train_vectors, val_vectors = blanket(data, val_data)
    training_acc_grid = np.zeros((len(γs), len(cs)))
    val_acc_grid     = np.zeros((len(γs), len(cs)))
    
    if fixed_c:
        supports = np.zeros(len(γs))
        for (i, γ) in enumerate(γs):
            svm = SVC(kernel="rbf", gamma=γ, C=10)
            svm.fit(train_vectors, y_train)
            supports[i] = sum(svm.n_support_)
        return supports
    
    elif fixed_γ:
        supports = np.zeros(len(cs))
        for (i, c) in enumerate(cs):
            svm = SVC(kernel="rbf", gamma=0.1, C=c)
            svm.fit(train_vectors, y_train)
            supports[i] = sum(svm.n_support_)
        return supports
    
    else:
        for (j, γ) in enumerate(γs):
            for (i, c) in enumerate(cs):
                clf = SVC(kernel="rbf", gamma=γ, C=c)
                clf.fit(train_vectors, y_train)
                training_acc_grid[len(γs)-j-1, i] = clf.score(train_vectors, y_train)
                val_acc_grid[len(γs)-j-1, i] = clf.score(val_vectors, y_val)     
        return training_acc_grid, val_acc_grid

#######################################
# Plotting functions
#######################################

def plot_support(support, cs, name="", axis="C"):
    fig, ax = plt.subplots()
    plt.plot(np.log10(cs), support)
    plt.xlabel(f"log({axis})")
    plt.ylabel("# of support vectors")
    plt.title(name)
    plt.savefig(name, bbox_inches='tight')
    plt.show()

def plot_acc(acc, cs, name=""):
    fig, ax = plt.subplots()
    plt.plot(np.log10(cs), acc)
    plt.xlabel("log(C)")
    plt.ylabel("Accuracy")
    plt.title(name)
    plt.savefig(name, bbox_inches='tight')
    plt.show()
    
def plot_heatmap(grid, γs, cs, name=""):
    temp_γs = γs[::-1]
    fig, ax = plt.subplots(figsize=(10, 6))
    heat_map = sns.heatmap(grid, linewidth = 1 , cbar = True, xticklabels=np.log10(cs), yticklabels=np.log10(temp_γs), annot=True, fmt=".3f")
    plt.xlabel("log(c)")
    plt.ylabel("log(γ)")
    plt.savefig(name)
    plt.title(name)
    plt.show()

#######################################
# implementation
#######################################
# import data
data     = pd.read_csv("IA3-train.csv")
val_data = pd.read_csv("IA3-dev.csv")

top_words_tfid_positive = preprocessing_data(data)[0]
top_words_tfid_negative = preprocessing_data(data, sentiment=False)[0]
top_words_common_positive = preprocessing_data(data, countvectorized=True)[0]
top_words_common_negative = preprocessing_data(data, countvectorized=True, sentiment=False)[0]

print(top_words_tfid_positive)
print(top_words_tfid_negative)
print(top_words_common_positive)
print(top_words_common_negative)

#######################################
# linear
#######################################
cs = [10**i for i in range(-4,5)]
acc_linear, support_linear = linear_validate(data, val_data, cs)

plot_acc(acc_linear, cs, "Accuracy plot for Linear")
plot_support(support_linear, cs, "Support vector for Linear")

cs_linear_new = [10**i for i in np.linspace(-1, 1, 10)]
acc_linear_new, support_linear_new = linear_validate(data, val_data, cs_linear_new )

plot_acc(acc_linear_new, cs_linear_new, "Close-up Linear score")

#######################################
# polynomial
#######################################
scores_quad, support_quad = quad_validate(data, val_data, cs)

plot_acc(scores_quad, cs, "Quad score")
plot_support(support_quad, cs, "Support vector for Quad")

#######################################
# rbf
#######################################

γs = [10**i for i in range(-5, 1)]
train_scores, val_scores = rbf_validate(data, val_data, cs, γs)

plot_heatmap(train_scores, γs, cs, "training accuracy map")

plot_heatmap(val_scores, γs, cs, "validation accuracy map")

fixed_c = rbf_validate(data, val_data, cs, γs, fixed_c=True)
plot_support(fixed_c, γs, axis="γ", name="Support vector for RBF with fixed c")

fixed_γ = rbf_validate(data, val_data, cs, γs, fixed_γ=True)
plot_support(fixed_γ, cs, name="Support vector for RBF with fixed γ")

