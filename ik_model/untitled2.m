clc
clear
pic_num = 1;

for i = 1:100
    
    plot(1:100,1:100);
    hold on
    plot(i,i,'o');
    F=getframe(gcf);
    I=frame2im(F);
    [I,map]=rgb2ind(I,256);

    if pic_num == 1
    imwrite(I,map,'test.gif','gif','Loopcount',inf,'DelayTime',0.1);

    else
    imwrite(I,map,'test.gif','gif','WriteMode','append','DelayTime',0.1);

    end

    pic_num = pic_num + 1;
    hold off
end
