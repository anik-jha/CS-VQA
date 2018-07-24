% Create the phiTphiX imagenet dataset both train and val


tic
mkdir('imagenet_validation_hadamard_0_01')
a = dir(fullfile('/home/kkulkar1/ImageNet/ILSVRC2012_val','*.JPEG'));
load hadamard_parameters_0_01
c = makecform('cmyk2srgb');

siz = [256 256]; %Size of the image. ONLY POWERS OF 2 ARE ALLOWED
M = 16384; 
size(row_perm)

fSl = @(x) x(row_perm);
fSlAdj = @(x) full(sparse(row_perm, 1, x, N, 1));
fSc = @(x) vec(x(col_perm));
fScAdj = @(x) x(rev_col_perm);

fPhi = @(x) fSl(fastWHtrans(fSc(x)));
fPhiAdj = @(x) fScAdj(fastWHtrans(fSlAdj(x)));

base_folder = '/home/slohit/codes/cs_deep_infer/imagenet/imagenet_validation_hadamard_0_01';
data_folder = '/home/kkulkar1/ImageNet/ILSVRC2012_val';

load max_min_hada_0_25.mat

% % Make training set
for i = 1:length(a)
        try
            im = imresize(double(imread(strcat(data_folder,'/',a(i).name))),[256 256]);
        catch
            ;
        end
        if (size(im,3) < 3)
            im = cat(3, im, im, im);
        end
        
        if (size(im,3) > 3)
            im = applycform(double(im), c);
        end
        
        im_r = im(:,:,1);
        im_g = im(:,:,2);           
        im_b = im(:,:,3);
        
        im_r_cs = reshape(fPhiAdj(fPhi(im_r(:))),[256 256]);
        im_g_cs = reshape(fPhiAdj(fPhi(im_g(:))),[256 256]);
        im_b_cs = reshape(fPhiAdj(fPhi(im_b(:))),[256 256]);
        
        
        im_r_cs = (255/(max_r_cs - min_r_cs))*(im_r_cs - min_r_cs);
        im_g_cs = (255/(max_g_cs - min_g_cs))*(im_g_cs - min_g_cs);
        im_b_cs = (255/(max_b_cs - min_b_cs))*(im_b_cs - min_b_cs);
        
               
        im_cs = uint8(cat(3, im_r_cs, im_g_cs, im_b_cs));
        imwrite( im_cs, strcat(base_folder, '/', a(i).name));
        
    a(i).name
end
toc
% output is phiTphiX imagenet dataset
