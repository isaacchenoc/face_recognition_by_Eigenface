% By Isaac Chen in 2019, with the inspiration from Prof. Hongdong Li, ANU

function main()
r = 231; 
c = 195;
k=10; %top-k eigenfaces
testno=10; 
matrix_col = 135;
test_col = 10;
the_matrix = [];
the_test = [];
for i=1:15
    imname = "Yale-FaceA/trainingset/subject";
    if i<10
        imname = imname+"0"+string(i)+".";
    else
        imname = imname+string(i)+".";
    end
    imname1= imname+"centerlight.png";
    imname2= imname+"leftlight.png";
    imname3= imname+"noglasses.png";
    imname4= imname+"normal.png";
    imname5= imname+"rightlight.png";
    imname6= imname+"sad.png";
    imname7= imname+"sleepy.png";
    imname8= imname+"surprised.png";
    imname9= imname+"wink.png";
    im1=imread(imname1);
    im2=imread(imname2);
    im3=imread(imname3);
    im4=imread(imname4);
    im5=imread(imname5);
    im6=imread(imname6);
    im7=imread(imname7);
    im8=imread(imname8);
    im9=imread(imname9);
    vec1=im1(:);
    vec2=im2(:);
    vec3=im3(:);
    vec4=im4(:);
    vec5=im5(:);
    vec6=im6(:);
    vec7=im7(:);
    vec8=im8(:);
    vec9=im9(:);
    the_matrix(:,end+1)=vec1;
    the_matrix(:,end+1)=vec2;
    the_matrix(:,end+1)=vec3;
    the_matrix(:,end+1)=vec4;
    the_matrix(:,end+1)=vec5;
    the_matrix(:,end+1)=vec6;
    the_matrix(:,end+1)=vec7;
    the_matrix(:,end+1)=vec8;
    the_matrix(:,end+1)=vec9;
end

imt1 = imread('Yale-FaceA/testset/subject01.glasses.png');
imt2 = imread('Yale-FaceA/testset/subject02.happy.png');
imt3 = imread('Yale-FaceA/testset/subject03.happy.png');
imt4 = imread('Yale-FaceA/testset/subject04.glasses.png');
imt5 = imread('Yale-FaceA/testset/subject05.happy.png');
imt6 = imread('Yale-FaceA/testset/subject06.happy.png');
imt7 = imread('Yale-FaceA/testset/subject07.glasses.png');
imt8 = imread('Yale-FaceA/testset/subject08.happy.png');
imt9 = imread('Yale-FaceA/testset/subject09.happy.png');
imt10 = imread('Yale-FaceA/testset/subject10.happy.png');
vect1=imt1(:);
vect2=imt2(:);
vect3=imt3(:);
vect4=imt4(:);
vect5=imt5(:);
vect6=imt6(:);
vect7=imt7(:);
vect8=imt8(:);
vect9=imt9(:);
vect10=imt10(:);
the_test(:,end+1)=vect1;
the_test(:,end+1)=vect2;
the_test(:,end+1)=vect3;
the_test(:,end+1)=vect4;
the_test(:,end+1)=vect5;
the_test(:,end+1)=vect6;
the_test(:,end+1)=vect7;
the_test(:,end+1)=vect8;
the_test(:,end+1)=vect9;
the_test(:,end+1)=vect10;

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

A=zeros(N,M);
for i=1:M
    A(:,i)=the_matrix(:,i)-x_mean;
end
AT=transpose(A);
L=AT*A;
[vs,~]=eigs(L,k);
for i=1:k
    k_eigenfaces(:,i)=A*vs(:,i);
end

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

figure();
subplot(2,5,1);imagesc(top_k_im{1});
colormap(gray);
subplot(2,5,2);imagesc(top_k_im{2});
colormap(gray);
subplot(2,5,3);imagesc(top_k_im{3});
colormap(gray);
subplot(2,5,4);imagesc(top_k_im{4});
colormap(gray);
subplot(2,5,5);imagesc(top_k_im{5});
colormap(gray);
subplot(2,5,6);imagesc(top_k_im{6});
colormap(gray);
subplot(2,5,7);imagesc(top_k_im{7});
colormap(gray);
subplot(2,5,8);imagesc(top_k_im{8});
colormap(gray);
subplot(2,5,9);imagesc(top_k_im{9});
colormap(gray);
subplot(2,5,10);imagesc(top_k_im{10});
colormap(gray);

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

bs=top_threeface(testno,:); 

reconstructed_x1=x_mean;
reconstructed_x2=x_mean;
reconstructed_x3=x_mean;
b1=B{bs(1)};
b2=B{bs(2)};
b3=B{bs(3)};
for i=1:k
    ui=k_eigenfaces(:,i);
    reconstructed_x1=reconstructed_x1+b1(i)*ui;
    reconstructed_x2=reconstructed_x2+b2(i)*ui;
    reconstructed_x3=reconstructed_x3+b3(i)*ui;
end

original_vec1=the_matrix(:,bs(1));
original_vec2=the_matrix(:,bs(2));
original_vec3=the_matrix(:,bs(3));
testvec=the_test(:,testno);
reconstructed_face1=vec2face(reconstructed_x1);
reconstructed_face2=vec2face(reconstructed_x2);
reconstructed_face3=vec2face(reconstructed_x3);
original_face1=vec2face(original_vec1);
original_face2=vec2face(original_vec2);
original_face3=vec2face(original_vec3);
testface=vec2face(testvec);
figure();
subplot(2,2,1);
imagesc(testface);
title("test");
colormap(gray)
subplot(2,2,2);
imagesc(original_face1);
title("matching image top 1");
colormap(gray)
subplot(2,2,3);
imagesc(original_face2);
title("matching image top 2");
colormap(gray)
subplot(2,2,4);
imagesc(original_face3);
title("matching image top 3");
colormap(gray)
%{
figure();
imagesc(reconstructed_face1);
colormap(gray)
figure();
imagesc(reconstructed_face2);
colormap(gray)
figure();
imagesc(reconstructed_face3);
colormap(gray)
%}
end

function fact_to_show = vec2face(vec)
r = 231;
c = 195;
fact_to_show = zeros(r,c);
for i=1:c
    st = (i-1)*r+1;
    ed = i*r;
    fact_to_show(:,i)=vec(st:ed);
end
end
