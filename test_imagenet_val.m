clear
close all
clc

%% Select measurement rate (MR): 0.25, 0.10, 0.04 or 0.01
mr = '0.10';
mr_str = mr(3:end);

fprintf('Measurement Rate = %s \n', mr)

%% Initializations

% Initialize Caffe
%addpath(genpath('../../../matlab'))
%addpath(genpath('/usr/local/MATLAB/R2017a/bin/'))
%addpath(genpath('/media/ssd/lichi/VQA1_0_01/codes'))
%addpath(genpath('/home/mayank/caffe_kuldeep/caffe/matlab'))
addpath(genpath('/home/anik/caffe/matlab'))

% Prototxt file for the selected MR
 prototxt_file = ['/media/ssd/lichi/VQA1_0_01/codes/deploy_prototxt_files/reconnet_0_', mr_str, '.prototxt']

    
    
 % Caffemodel for the selected MR
caffemodel = ['/media/ssd/lichi/VQA1_0_01/codes/caffemodels/reconnet_0_', mr_str, '.caffemodel']
caffe.set_mode_gpu();
caffe.set_device(0);
%imagenet root

fileID = fopen('output_val.txt','w');
c=makecform('cmyk2srgb');



% Dataset for testing
test_images_dir = '/media/twotb/imagenet/ILSVRC2012_val';
%test_images_dir = test_dirs(testdir).name;
test_images = dir(fullfile(test_images_dir,'*.JPEG'));

%test_images = test_images(3:end);

%output_dir = ['./reconstruction_results/mr_0_', mr_str, '/'];
output_dir = ['./reconstruction_results/mr_0_', mr_str, '_val/'];
mkdir(output_dir)

% Load the measurement matrix for the selected MR
load(['phi_0_', mr_str, '_1089.mat']);

psnr = zeros(11,1);
time_complexity = zeros(11,1);

%%
for image_number = 1:length(test_images)
    try
        caffe.reset_all();
    catch
        caffe.reset_all();
    end
    image_name = test_images(image_number).name;
    %input_im_nn = im2double(imread(fullfile(test_images_dir,image_name))); %Input for the ReconNet
    % catch the error : JPEG imamyfileges with CMYK colorspace are not currently supported.
    [~,name,~] = fileparts(image_name);
    outname=[output_dir name '.jpg'];
    if exist(outname,'file')
        fprintf('file existed')
    else
    %input_im_nn = im2double(imread(fullfile(test_images_root,test_images_dir,image_name))); %Input for the ReconNet
    
    try
	fullfile(test_images_dir,image_name);
        im = im2double(imread(fullfile(test_images_dir,image_name))); %Input for the ReconNet
        
        if (size(im,3)<3)
            im=cat(3,im,im,im);
        end
        if(size(im,3)>3)
            im=applycform(im,c);
        end
        input_im_nn= imresize(im,[256,256]);
        ch = "Resize";
        
        block_size = 33;
        num_blocks = ceil(size(input_im_nn,1)/block_size)*ceil(size(input_im_nn,2)/block_size)
       
        %modify_prototxt(prototxt_file, num_blocks);
        net = caffe.Net(prototxt_file, caffemodel, 'test');
        
        % Determine the size of zero pad
        %[row, col] = size(input_im_nn);
        [row, col, ch] = size(input_im_nn); % for color image
        if mod(row,block_size) ==0
            row_pad=0;
            
        else
            row_pad = block_size-mod(row,block_size);
        end
        if mod(col,block_size)==0
            col_pad=0;
        else
            col_pad = block_size-mod(col,block_size);
        end
        % Do zero padding
        im_pad_nn = [input_im_nn, zeros(row,col_pad,ch)];  % zeros(row,col_pad,ch) for color image
        im_pad_nn = [im_pad_nn; zeros(row_pad,col+col_pad,ch)]; % for color image
        
        
        for k = 1:ch
        count = 0;
        for i = 1:size(im_pad_nn,1)/block_size
            for j = 1:size(im_pad_nn,2)/block_size
                % Access the (i,j)th block of image 
                ori_im_nn = im_pad_nn((i-1)*block_size+1:i*block_size,(j-1)*block_size+1:j*block_size,k);
                count = count + 1;
                %CSCNN - Take the compressed measurements of the block
                y = phi*ori_im_nn(:);
                input_deep(count,1,:,1) = y;
            end
        end
        
        start_time = tic;
        net = caffe.Net(prototxt_file, caffemodel, 'test');
        % input_deep contains the set of CS measurements of all block,
        % net.forward compute reconstructions of all blocks parallelly
        temp = net.forward({permute(input_deep,[4 3 2 1])});
        
        %Rearrange the reconstructions to form th[~,name,~] = fileparts(image_name);e final image im_comp_cscnn
        count = 0;
        
        for i = 1:size(im_pad_nn,1)/block_size
            for j = 1:size(im_pad_nn,2)/block_size
                count = count + 1;
                im_comp((i-1)*block_size+1:i*block_size,(j-1)*block_size+1:j*block_size,k) = temp{1}(:,:,1,count);
            end
        end
        clear temp
        end
        time_complexity(image_number) = toc(start_time);

        rec_im = im_comp(1:row, 1:col,:);
        [~,name,~] = fileparts(image_name);
        imwrite(rec_im, [output_dir name '.jpg']);
        
        %imshow(rec_im)
        %pause;[~,name,~] = fileparts(image_name);
       
        diff = input_im_nn - rec_im;
        sig = sqrt(mean(input_im_nn(:).^2));
        diff = diff(:);
        rmse = sqrt(mean(diff(:).^2));
        psnr(image_number) = 20*log10(1/rmse);
        
        %clear im_comp temp input_deep
        clear im_comp input_deep 
        fprintf(fileID,'\n %15s: PSNR = %f dB, Time = %f  seconds\n', image_name, psnr(image_number), time_complexity(image_number));
        %fprintf('\n %15s: PSNR = %f dB, Time = %f  seconds\n', image_name, psnr(image_number), time_complexity(image_number));
    catch
         fprintf(fileID,'there is error');
    end
        
    end
end


fprintf(fileID,['\n All reconstruction results are saved in ', output_dir, '\n']);

fprintf(fileID,['\n All reconstruction results are saved  ', '\n']);
fclose(fileID);
