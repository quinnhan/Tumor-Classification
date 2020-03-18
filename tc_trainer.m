function [result,w,U,S,V,th,sorttumor,sortcnotumor]=tc_trainer(tumor,notumor,feature) 

nt=length(tumor(1,:));
nnc=length(notumor(1,:));

[U,S,V]=svd([tumor,notumor],0);

images = S*V';
U = U(:,1:feature);
tumors = images(1:feature,1:nt);
notumors = images(1:feature,nt+1:nt+nnc);

md = mean(tumors,2);
mc = mean(notumors,2);

Sw=0;
for i=1:nt
    Sw = Sw + (tumors(:,i)-md)*(tumors(:,i)-md)';
end
for i=1:nnc
    Sw = Sw + (notumors(:,i)-mc)*(notumors(:,i)-mc)';
end

Sb = (md-mc)*(md-mc)';

[V2,D] = eig(Sb,Sw);  % linear discriminant analysis
dd = diag(D);
[lambda,ind] = max(abs(dd));
w = V2(:,ind);
w = w/norm(w,2);

vtumor = w'*tumors;
vnotumor = w'*notumors;

result = [vtumor,vnotumor];

if mean(vtumor)>mean(vnotumor) 
    w = -w;
    vtumor = -vtumor;
    vnotumor = -vnotumor;
end
% dog < threshold < cat

sorttumor = sort(vtumor);
sortcnotumor = sort(vnotumor);

% figure(4)
% subplot(2,2,1)
% hist(sortdog,30); hold on, plot([18.22 18.h22],[0 10],'r')
% set(gca,'Xlim',[-200 200],'Ylim',[0 10],'Fontsize',[14]), title('dog')
% subplot(2,2,2)
% hist(sortcat,30,'r'); hold on, plot([18.22 18.22],[0 10],'r')
% set(gca,'Xlim',[-200 200],'Ylim',[0 10],'Fontsize',[14]), title('cat')
    
t1 = length(sorttumor);
t2 = 1;
 while sorttumor(t1)>sortcnotumor(t2)
    t1 = t1-1;
    t2 = t2+1;
end
th = (sorttumor(t1)+sortcnotumor(t2))/2;

