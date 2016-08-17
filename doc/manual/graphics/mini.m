\begin{Verbatim}[frame=single, fontsize=\small]
% imports
import model_io.*
import data_io.*
import render.*
% end of imports




% read model and MNIST test data
nn = model_io.read(<model_path>);
X = data_io.read(<data_path>);
% pick first image, normalize to [-1 1]
X = X(1,:) / 127.5 - 1; 

% forward pass through network
Ypred = nn.forward(X); 
% apply lrp to explain prediction of X
R = nn.lrp(Ypred); 

% render rgb images and save as image
digit = render.digit_to_rgb(X);
% render heatmap R, use X as outline
hm = render.hm_to_rgb(R,X); 
render.save_image({digit,hm},<i_path>);
\end{Verbatim}




