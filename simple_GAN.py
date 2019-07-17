###########################################################
#    Project:     Implementation of GAN using XOR gate
#    Created by:  Hardik Ajmani
#    Proposal by: Mrinal Wahal
#    Date:        17th July 2019
########################################################### 
import numpy as np

def sigmoid(x, derivative = False):
    if derivative == True: return x(1-x)
    return (1/(1 + np.exp(-x)))


def think(input, layer_1, layer_2, bias_1, bias_2):
    l1 = sigmoid(np.dot(input, layer_1) + bias_1)
    l2 = sigmoid(np.dot(l1, layer_2) + bias_2)
    return l2

def to_binary(mat):
    mat[mat > 0.5] = 1
    mat[mat < 0.5] = 0

    return mat


#data 
X = np.array([[0,0,0], [0,0,1], [0,1,0], [0,1,1], [1,0,0], [1,0,1], [1,1,0], [1,1,1]])
y = np.array([[0, 1, 1, 0, 1, 0, 0, 1]])


#Defining discriminator

#dimensions
#   X(input) = 8*3 (8 examples)
#   Layer_1  = 3*6
#   Layer_2  = 6*1
#   Output = 0/1

#initalizing weights and bias for discriminator
weights_layer_1 = np.random.random((3 , 6))
weights_layer_2 = np.random.random((6 , 1))

bias_layer_1 = np.random.random((8, 6))
bias_layer_2 = np.random.random((8, 1))

def disc_train(input, output, layer_1, layer_2, bias_1, bias_2):
    
    m = len(input) #number of examples
    alpha = 0.001  #learning rate

    for i in range(10000):
        #forward propogation
        l1 = sigmoid(np.dot(input, layer_1) + bias_1)            #(input)8*3 X (layer_1)3*6 = (l1)8*6 + (b1)8*6
        l2 = sigmoid(np.dot(l1, layer_2) + bias_2)               #(l1)8*6 X (layer2)6*1 = (l2)8*1 + (b2)8*1

        loss = -np.sum(np.dot(output, np.log(l2)) + np.dot(1 - output, np.log(1 - l2))) / m

        #back propogation
        delta_z2 = np.subtract(l2,output.T)                           #(delta_z2)8*1
        #print(np.shape(delta_z2))
        delta_w2 = np.dot(l1.T, delta_z2) / m                       #(delta_w2)6*1
        delta_b2 = np.sum(delta_z2, axis= 1, keepdims=True) / m     #(delta_b2)8*1            

        delta_z1 = np.dot(delta_z2, layer_2.T) * l1                 #(delta_z1)8*6     
        delta_w1 = np.dot(input.T, delta_z2) / m                    #(delta_w1)3*6
        delta_b1 = np.sum(delta_z1, axis= 1, keepdims=True) / m     #(delta_b1)8*6

        layer_2 -= alpha * delta_w2
        layer_1 -= alpha * delta_w1

        bias_1 -= alpha * delta_b1
        bias_2 -= alpha * delta_b2 

        #if i%100 == 0 :  print("At iteration: {0},  Loss: {1}".format(i, loss))

    return (layer_1, layer_2, bias_1, bias_2)

print("Before training")
print(to_binary(think(X, weights_layer_1, weights_layer_2, bias_layer_1, bias_layer_2)))

print("Training Discriminator")
disc_weights_layer_1, disc_weights_layer_2, disc_bias_layer_1, disc_bias_layer_2 = disc_train(X, y, weights_layer_1, weights_layer_2, bias_layer_1, bias_layer_2)

print("After training")
print(to_binary(think(X, disc_weights_layer_1, disc_weights_layer_2, disc_bias_layer_1, disc_bias_layer_2)))



#Defining generator

#dimensions
#   Z(random input) = 8*3 (8 examples)
#   Layer_1         = 3*6
#   Layer_2         = 6*3
#   Output          = 3*1

#initializing Z (random noise)
Z = np.random.random((8, 3))  #like [0.3, 0.2, 0.1] * 8 examples
Y = np.array([[1, 1, 1, 1, 1, 1, 1, 1]])
#initalizing weights and bias for generator
weights_layer_1 = np.random.random((3 , 6))
weights_layer_2 = np.random.random((6 , 3))

bias_layer_1 = np.random.random((8, 6))
bias_layer_2 = np.random.random((8, 3))

print("Before training generative network")
print(to_binary(think(Z, weights_layer_1, weights_layer_2, bias_layer_1, bias_layer_2)))


def gen_train(input, output, layer_1, layer_2, bias_1, bias_2):
    alpha = 0.09
    m = len(input)
    for i in range(10000):
        #forward propogation
        l1_Gen      = np.dot(input, layer_1) + bias_1                                      #(input)8*3 X (layer_1)3*6 = (l1)8*6 + (b1)8*6
        l1_Gen_sigm = sigmoid(l1_Gen)
        l2_Gen      = np.dot(l1_Gen_sigm, layer_2) + bias_2                                      #(l1)8*6 X (layer2)6*1 = (l2)8*1 + (b2)8*1
        l2_Gen_sigm = sigmoid(l2_Gen)

        #sending the generated [1 , 0, 0] output to trained layers of Discriminator
        l1_Disc      = np.dot(l2_Gen_sigm, disc_weights_layer_1) + disc_bias_layer_1                 #(l2_Gen)8*3 X (disc_layer_1)3*6 = (l1_Disc)8*6 + (b1)8*6
        l1_Disc_sigm = sigmoid(l1_Disc) 
        l2_Disc      = np.dot(l1_Disc_sigm, disc_weights_layer_2) + disc_bias_layer_2           #(l1_Disc)8*6 X (disc_layer_2)6*1 = (l2_Disc)8*1 + (b2)8*1
        l2_Disc_sigm = sigmoid(l2_Disc)

        loss = -np.sum(np.log(1 - l2_Disc_sigm)) / m

        #Backpropogation
        delta_z2_a = ((-1/ l2_Disc_sigm) * np.dot(sigmoid(l2_Disc), disc_weights_layer_2.T) * l1_Disc_sigm).dot(disc_weights_layer_1.T)
        delta_z2_b = sigmoid(l2_Gen)
        delta_z2_c = l1_Gen_sigm
        delta_w2   = np.dot(delta_z2_c.T, delta_z2_a * delta_z2_b)
        delta_b2   = delta_z2_a * delta_z2_b

        delta_z1_a = np.dot(delta_z2_a * delta_z2_b, layer_2.T)
        delta_z1_b = sigmoid(l1_Gen)
        delta_z1_c = input
        delta_w1   = np.dot(delta_z1_c.T, delta_z1_a * delta_z1_b)
        delta_b1   = delta_z1_a * delta_z1_b

        layer_1 -= alpha * delta_w1
        layer_2 -= alpha * delta_w2
        bias_1  -= alpha * delta_b1
        bias_2  -= alpha * delta_b2

        if i%1000 == 0 :  print("At iteration: {0},  Loss: {1}".format(i, loss))

    return(layer_1, layer_2, bias_1, bias_2)


gen_weights_layer_1, gen_weights_layer_2, gen_bias_layer_1, gen_bias_layer_2 = gen_train(Z, Y, weights_layer_1, weights_layer_2, bias_layer_1, bias_layer_2)

print("After training generative network")
print(to_binary(think(Z, gen_weights_layer_1, gen_weights_layer_2, gen_bias_layer_1, gen_bias_layer_2)))
