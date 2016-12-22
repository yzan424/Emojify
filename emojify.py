import json
from nltk import MaxentClassifier
from math import exp, pow, log, sqrt
from collections import defaultdict

def ScanTraining(training, emojis,counts):
    mapped_training = defaultdict(list)
    emojis_training = defaultdict()
    print len(training)

    for curr in training:
        curr_emojis = set()
        if curr[:4] == "RT @":
            for i in range(4,len(curr) - 1):
                if curr[i] == " ":
                    curr = curr[i + 1:]
                    break
        i = 0
        while i < len(curr) - 1:
            if curr[i] == " " and curr[i+1] == " ":
                curr = curr[:i] + curr[i+1:]
            # if curr[i] > u'\ud800' and curr[i] < u'\udbff' and curr[i+1] > u'\udc00' and curr[i+1] < u'\udfff':
            # if (curr[i] == u'\ud83d' and curr[i+1] == u'\ude0a') or (curr[i] == u'\ud83d' and curr[i+1] == u'\ude12'):
            if curr[i] == u'\ud83d' and curr[i+1] >= u'\ude00' and curr[i+1] <= u'\ude13':
                if curr[i:i+2] not in emojis:
                    emojis[curr[i:i+2]] = defaultdict(int)
                
            if curr[i] > u'\ud800' and curr[i] < u'\udbff' and curr[i+1] > u'\udc00' and curr[i+1] < u'\udfff':
                curr_emojis.add(curr[i:i+2])
                curr = curr[:i] + curr[i+2:]
                i -= 1
                # if i != len(curr) - 2 and curr[i+2] != " ":
                #     curr = curr[:i] + " " + curr[i+2:]
                # if i != 0 and curr[i-1] != " ":
                #     curr = curr[:i] + " " + curr[i:]
            i += 1

        curr = curr.lower().split(" ")
        for emoji in curr_emojis:
            # key emoji is still in curr
            # if curr[i] > u'\ud800' and curr[i] < u'\udbff' and curr[i+1] > u'\udc00' and curr[i+1] < u'\udfff':
            # if (emoji == u'\ud83d\ude0a') or (emoji == u'\ud83d\ude12'):
            if emoji >= u'\ud83d\ude00' and emoji <= u'\ud83d\ude13':
                mapped_training[emoji].append(curr)
                for word in curr:
                # if word != emoji:
                    if not ((word > u'\ud800' and word < u'\udbff') or (word > u'\udc00' and word < u'\udfff')):
                        emojis[emoji][word.lower()] += 1
            if str(curr) not in emojis_training:
                emojis_training[str(curr)] = set()
            emojis_training[str(curr)].add(emoji)

            

    for emoji in emojis.keys():
        unique_words = set()
        for word in emojis[emoji].keys():
            # if word != emoji:
            if not ((word > u'\ud800' and word < u'\udbff') or (word > u'\udc00' and word < u'\udfff')):
                unique_words.add(word.lower())
        for word in unique_words:
            counts[word] += 1

    return mapped_training,emojis_training

def ScanTesting(testing):
    mapped_testing = defaultdict(list)
    emojis_testing = defaultdict()
    print len(testing)
    for curr in testing:
        curr_emojis = set()
        if curr[:4] == "RT @":
            for i in range(4,len(curr) - 1):
                if curr[i] == " ":
                    curr = curr[i + 1:]
                    break
        i = 0
        while i < len(curr) - 1:
            if curr[i] == " " and curr[i+1] == " ":
                curr = curr[:i] + curr[i+1:]
            if curr[i] > u'\ud800' and curr[i] < u'\udbff' and curr[i+1] > u'\udc00' and curr[i+1] < u'\udfff':
                curr_emojis.add(curr[i:i+2])
                curr = curr[:i] + curr[i+2:]
                i -= 1
                # if i != len(curr) - 2 and curr[i+2] != " ":
                #     curr = curr[:i+2] + " " + curr[i+2:]
                # if i != 0 and curr[i-1] != " ":
                #     curr = curr[:i] + " " + curr[i:]
            i += 1

        curr = curr.lower().split(" ")
        for emoji in curr_emojis:
            # key emoji is still in curr
            # if curr[i] > u'\ud800' and curr[i] < u'\udbff' and curr[i+1] > u'\udc00' and curr[i+1] < u'\udfff':
            # if (emoji == u'\ud83d\ude0a') or (emoji == u'\ud83d\ude12'):
            if emoji >= u'\ud83d\ude00' and emoji <= u'\ud83d\ude13':
                mapped_testing[emoji].append(curr)
            if str(curr) not in emojis_testing:
                emojis_testing[str(curr)] = set()
            emojis_testing[str(curr)].add(emoji)

    # print len(mapped_testing)
    return mapped_testing,emojis_testing


# gets fucked when you only have two classes cuz all words in query get 0 weight if appear in both classes
# some tweets have multiple emojis, so belongs to multiple classes despite havign EXACT SAME feature values
# append features to one another versus testing separately
def CosineSimilarity(documents,query,counts):
    query_vector = defaultdict(int)
    query = query.split(" ")
    for word in query:
        query_vector[word] += 1
    emoji_cosine_map = defaultdict(float)
    for emoji in documents.keys():
        numerator = 0.0
        denominator1 = 0.0
        denominator2 = 0.0
        maxword = ""
        maxnum = 0
        for word in documents[emoji].keys():
            numerator += documents[emoji][word] * query_vector[word] * log(len(documents.keys())/float(counts[word])) * log(len(documents.keys())/float(counts[word]))
            if query_vector[word]:
                print word,documents[emoji][word],query_vector[word],log(len(documents.keys())/float(counts[word]))
            denominator1 += documents[emoji][word] * documents[emoji][word] * log(len(documents.keys())/float(counts[word])) * log(len(documents.keys())/float(counts[word]))
            if maxnum < documents[emoji][word] * log(len(documents.keys())/float(counts[word])):
                maxnum = documents[emoji][word] * log(len(documents.keys())/float(counts[word]))
                maxword = word
        print emoji,maxword,maxnum
        emoji_cosine_map[emoji] = numerator / sqrt(denominator1)

    max_cosine = 0
    max_emoji = documents.keys()[0]
    for emoji in emoji_cosine_map.keys():
        print emoji,emoji_cosine_map[emoji]
        if emoji_cosine_map[emoji] > max_cosine:
            max_cosine = emoji_cosine_map[emoji]
            max_emoji = emoji

    print max_emoji,max_cosine

def MaxEnt(mapped_training,mapped_testing,emojis_training,emojis_testing):
    maxent_pairlist = []
    for emoji in mapped_training.keys():
        for tweet in mapped_training[emoji]:
            feature_dict = defaultdict(int)
            for word in tweet:
                feature_dict[word] = 1
            #COMMENT THIS PART OUT TO REMOVE EMOJIS
            # for other_emoji in emojis_training[str(tweet)]:
            #     if other_emoji != emoji:
            #         feature_dict[other_emoji] = 1
            maxent_pairlist.append((feature_dict,emoji))
    
    test_list = []
    for emoji in mapped_testing.keys():
        for tweet in mapped_testing[emoji]:
            param_dict = defaultdict(int)
            other_emojis = set()
            for word in tweet:
                # if word != emoji:
                param_dict[word] = 1
            for other_emoji in emojis_testing[str(tweet)]:
                if other_emoji != emoji:
                    param_dict[other_emoji] = 1
                    other_emojis.add(other_emoji)
            test_list.append((tweet,param_dict,emoji,other_emojis))

    classifier = MaxentClassifier.train(maxent_pairlist,max_iter=10)
    acc = 0.0
    
    for test in test_list:
        class_dist = classifier.prob_classify(test[1])
        pred_dict = defaultdict(float)
        for pred in class_dist.samples():
            pred_dict[pred] = class_dist.prob(pred)
        max_emoji = max(pred_dict, key=pred_dict.get)
        while max_emoji in test[3]:
            del pred_dict[max_emoji]
            max_emoji = max(pred_dict, key=pred_dict.get)
        if max_emoji == test[2]:
            acc += 1.0
        print test[0],max_emoji
        
    print acc / len(test_list)

def main():
    # f = open('tweets.txt', 'a')
    # f.write(('"I just want to not be bothered \ud83d\ude02\ud83d\ude44"]}'))
    # f.close()
    f = open('tweets_20.txt', 'r')
    emojis = defaultdict()
    tweets = list(set(json.loads(f.read())["statuses"]))
    print len(tweets)
    # 5000, 8900, 20000
    training_tweets = tweets[:8900]
    testing_tweets = tweets[8900:]
    inverse_doc_freq = defaultdict(float)
    mapped_training,emojis_training = ScanTraining(training_tweets,emojis,inverse_doc_freq)
    mapped_testing,emojis_testing = ScanTesting(testing_tweets)
    # CosineSimilarity(emojis,"i am so irritated and angry".lower(),inverse_doc_freq)
    MaxEnt(mapped_training,mapped_testing,emojis_training,emojis_testing)

if __name__ == "__main__": 
    main()