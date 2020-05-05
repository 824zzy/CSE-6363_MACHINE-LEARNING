""" Description
Writea subroutine“distance(x1,x2)”to compute the distance between two data instances x1,x2. 
Inside distance(), you should group numerical attributes together as num-set, group categorical attribute together as cat-set, etc. 
On num-set, you use Euclidean distance. On cat-set, you use Hamming distance.
(For example, we have x1 = [11, “warm-blooded”, 2] and x2 = [13, “cold-blooded”, 3], distance(x1,x2) will be(11-13)2+ 1 + (2-3)2= 6.)
"""
def distance(x1, x2):
    # Input can be numer-set/group categorical attribute
    dist = 0
    for i in range(len(x1)):
        if str(x1[i]).isdigit():
            dist += (x1[i]-x2[i])**2
        else:
            max_len = max(len(x1[i]), len(x2[i])) if len(x1[i])!=len(x2[i]) else len(x1[i])
            w1, w2 = x1[i].ljust(max_len, '0'), x2[i].ljust(max_len, '0')
            dist += sum([1 if w1[j]!=w2[j] else 0 for j in range(max_len)])
    return dist
    
if __name__ == "__main__":
    # dist = distance([11, "warm-blooded", 2], [13, "cold-blooded", 3])
    dist = distance([11, "warm-blooded", 2], [13, "cold-blooded", 3])
    print(dist) # 9
    dist = distance(['small'], ['medium'])
    print(dist) # 6