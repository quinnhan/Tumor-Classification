function tcData = tc_wavelet(tcfile,rows,columns) 
[m,n]=size(tcfile); % 4096 x 80 

nbcol = size(colormap(gray),1);
for i=1:n
  X=double(reshape(tcfile(:,i),rows, columns)); 
  [cA,cH,cV,cD]=dwt2(X,'haar');
  cod_cH1 = wcodemat(cH,nbcol);
  cod_cV1 = wcodemat(cV,nbcol); 
  cod_edge=cod_cH1+cod_cV1; 
  tcData(:,i)=cod_edge(:);
end