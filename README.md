# face_recognition_by_Eigenface
Face recognition by Eigenface, implemented in matlab.

Acknowledgement: The my implementation was inspired by Prof. Hongdong Li, ANU

This is a really interesting algorithm for face recognition, although it might be a bit hard to understand at the beginning. To learn this algorithm, basic understanding about eigen vector, eigen value, covariance matrix and Principal component analysis (PCA) would be necessary. 

# Implementation

I used the dataset Yale-FaceA (a part of Yale-Face) in this imlementation (as it's quite small, I have included in this repo). By dividing the images into training set and testing set (already divided in the repo), I then have one 45045 by 135 matrix standing for all train images and one 45045 by 10 matrix standing for all test images.

Then I get the sum of all columns of the train image matrix and divided it by column size so that I get the main face (step 1 in PCA). Since at this moment, the mean face is still a vector, so I convert the vector back to a 231 by 195 image. 

```
[N,M] = size(the_matrix);
k_eigenfaces=zeros(N,k);
x_mean = sum(the_matrix,2)/M;

mean_face = zeros(r,c);
for i=1:c
    st = (i-1)*r+1;
    ed = i*r;
    mean_face(:,i)=x_mean(st:ed);
end
figure();
imagesc(mean_face);
colormap(gray)
```

And I get the mean face image.
![meanface](https://i.imgur.com/KQJzf7m.png)

Using the mean face, I can get the A (step 2 in PCA), which is a 45045 by 135 matrix containing the difference between each original image vector and the mean face vector.

```
A=zeros(N,M);
for i=1:M
    A(:,i)=the_matrix(:,i)-x_mean;
end
```

Given that A is a 45045 by 135 matrix, it is unrealistic to directly compute covariance matrix C=A * transpose(A) (step 3 in PCA). However, in order to get the eigenvectors and eigenvalues of C, we don’t necessarily need to compute C. 

According to Turk and Pentland's original paper on "EigenFace" published in 1991. We can first compute L=transpose(A) * A and use L to get the eigenvectors and eigenvalues of C. To compute L, it is not difficult since it is a 135 by 135 matrix coming from L=transpose(A) * A. Moreover, to calculate its eigenvectors and eigenvalues is also not different given its size. Consider v is an eigenvector of L and u is the eigenvalue, we have transpose(A) * A * v=u * v. If we multiply both sides by A, we have A * transpose(A) * A * v=u * A * v, where A * transpose(A) = C, which means 𝐴𝑣 is the eigenvector of C and u is the eigenvalue. So, using this way we can compute the eigenvectors and eigenvalues of C in a much faster way (step 4&5 in PCA).

```
AT=transpose(A);
L=AT*A;
[vs,~]=eigs(L,k);
for i=1:k
    k_eigenfaces(:,i)=A*vs(:,i);
end
```

Since when we compute the eigenvectors and eigenvalues of L we use the matlab built-in eigs(), so the eigenvectors for C we have are basically the top k principle components (eigenfaces) of C. Then, we can visualize these top k eigenfaces, which is similar with what we have done in visualizing mean face and we can get the eigenface images (k=10).

```
top_k_im={};
for j=1:k
    a_ef = k_eigenfaces(:,j);
    new_ef = zeros(r,c);
    for i=1:c
        st = (i-1)*r+1;
        ed = i*r;
        new_ef(:,i)=a_ef(st:ed);
    end
    top_k_im{end+1}=new_ef;
end
```

![eigenface](https://i.imgur.com/xliXZUP.png)

As we have the top k eigenfaces, we need to find out the coefficients of the eigenfaces for all train and test images (step 6 in PCA), since we want every image can be represented by a linear combination of the top k eigenfaces (the key idea of using eigenface for face recognition). since we know the image, the mean face and all eigenface, we can get every coefficient by b = ((x - x_mean) * u) / (u * u), considering u as a eigenface and b as its coefficient. 

So that we can get 135 k by 1vectors for train images and each vector containing k coefficients for the top k eigenfaces, and similarly 10 k by 1 vectors for test images. All of these vectors are stored in separate cell arrays (one for train images and one for test images).

```
B={};
for i=1:matrix_col
    x=the_matrix(:,i);
    dif_x= x-x_mean;
    bs=[];
    for j=1:k
        ui=k_eigenfaces(:,j);
        doc_xui=0;
        doc_ui=0;
        for d=1:r*c
            doc_xui=doc_xui+dif_x(d)*ui(d);
            doc_ui=doc_ui+ui(d)*ui(d);
        end
        bi=doc_xui/doc_ui;
        bs=[bs bi];
    end
    bs=transpose(bs);
    B{end+1}=bs;
end

B_test={};
for i=1:test_col
    x=the_test(:,i);
    dif_x= x-x_mean;
    bs=[];
    for j=1:k
        ui=k_eigenfaces(:,j);
        doc_xui=0;
        doc_ui=0;
        for d=1:r*c
            doc_xui=doc_xui+dif_x(d)*ui(d);
            doc_ui=doc_ui+ui(d)*ui(d);
        end
        bi=doc_xui/doc_ui;
        bs=[bs bi];
    end
    bs=transpose(bs);
    B_test{end+1}=bs;
end
```

Once we have these coefficients, given a test image, we can use a near-neighbour search (which to find out the train image coefficient vector that is closest to the test image vector) to find out the top 3 train images that match the test image the best. I also use a cell array to store the top 3 matching train images for all test images.

```
top_threeface = zeros(test_col,3);
for i=1:test_col
    top3id = [0 0 0];
    min_dist1 = 99999999997;
    min_dist2 = 99999999998;
    min_dist3 = 99999999999;
    for j=1:matrix_col
        dist=0;
        for d=1:k
            point1 = B{j};
            point2 = B_test{i};
            dist=dist+power((point1(d)-point2(d)),2);
        end
        dist=sqrt(dist);
        if dist<min_dist1
            min_dist3=min_dist2;
            min_dist2=min_dist1;
            min_dist1=dist;
            top3id(3)=top3id(2);
            top3id(2)=top3id(1);
            top3id(1)=j;
        elseif dist<min_dist2
            min_dist3=min_dist2;
            min_dist2=dist;
            top3id(3)=top3id(2);
            top3id(2)=j;
        elseif dist<min_dist3
            min_dist3=dist;
            top3id(3)=j;
        end
    end
    top_threeface(i,:)=top3id;
end
```

As I use the eigenface face recognizer to find the top-3 matching images for some test images, the result is as below.
The performance overall is acceptable. Although there are algorithms that have higher accuracies, this algorithm is still fun to learn. 

![result](https://i.imgur.com/AhmiTeR.png)
![result](https://i.imgur.com/Fwm0zgc.png)
![result](https://i.imgur.com/x3vP3lx.png)


