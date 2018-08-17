function varargout = Training(varargin)
% TRAINING MATLAB code for Training.fig
%      TRAINING, by itself, creates a new TRAINING or raises the existing
%      singleton*.
%
%      H = TRAINING returns the handle to a new TRAINING or the handle to
%      the existing singleton*.
%
%      TRAINING('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in TRAINING.M with the given input arguments.
%
%      TRAINING('Property','Value',...) creates a new TRAINING or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before Training_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to Training_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help Training

% Last Modified by GUIDE v2.5 30-Apr-2018 17:44:48

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @Training_OpeningFcn, ...
                   'gui_OutputFcn',  @Training_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before Training is made visible.
function Training_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to Training (see VARARGIN)
axes(handles.axes2)
matlabImage = imread([pwd , '\images\player_record2.png']);
image(matlabImage)
axis off
axis image
axes(handles.axes1)
matlabImage = imread([pwd ,'\images\Satz.png']);
image(matlabImage)
axis off
axis image
axes(handles.axes12)
matlabImage = imread([pwd ,'\images\Bild1.png']);
image(matlabImage)
axis off
axis image
axes(handles.axes13)
matlabImage = imread([pwd ,'\images\Bild2.png']);
image(matlabImage)
axis off
axis image
global setper name VX
setper=0;
clearvars -global name 
VX=cell(50,1);
% Choose default command line output for Training
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);


% UIWAIT makes Training wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = Training_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in codebookbutton.
function codebookbutton_Callback(hObject, eventdata, handles)
% hObject    handle to codebookbutton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of codebookbutton


% --- Executes on button press in gmmbutton2.
function gmmbutton2_Callback(hObject, eventdata, handles)
% hObject    handle to gmmbutton2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of gmmbutton2


% --- Executes on button press in knnbpbutton3.
function knnbpbutton3_Callback(hObject, eventdata, handles)
% hObject    handle to knnbpbutton3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of knnbpbutton3


% --- Executes on button press in radiobutton4.
function radiobutton4_Callback(hObject, eventdata, handles)
% hObject    handle to radiobutton4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of radiobutton4


% --- Executes on button press in pushbutton7.
function pushbutton7_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton7 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global name VX setper
if get(handles.radiobutton5,'Value')==1
    codebook_size=2;
    while codebook_size<=str2num(get(handles.edit7,'String'))
        for u=1:setper
            clearvars -except u codebook_size VX name setper handles
            thr=0.999;
            distance_alt=inf;
            clear V1
            
            V1=cell2mat(VX(u));
            [num_mfcc num_frames] =size(V1);
            op=1;
            i=1;
            l=0;
            C(i,:)=sum(V1')/num_frames;
            num_code=size(C,1);
            while (num_code<codebook_size)
                %% Kreation vom Codebook
                num_code=size(C,1);
                for i=1:num_code
                    C(i+2^l,:)=C(i,:)+(0.5*rand(1,size(V1,1)));
                end
                l=l+1;
                num_code=size(C,1);
                dis_unterschied=0;
                %% Suchen von Codebook was am besten passt
                while (dis_unterschied<thr)
                    for k=1:num_frames
                        for x=1:num_code
                            d(x)= sum(abs(V1(:,k)'-C(x,:)));
                        end
                        c(k)=find(d == min(d(:)));
                    end
                    %%Unterschiede Berechnen
                    distance=0;
                     for k=1:num_frames
                         distance= sum(abs(V1(:,k)'-C(c(k),:)))+ distance;
                     end
                    distance=distance/num_frames;
                    %%Verbesserung der Codebooks
                    for x=1:num_code
                        rechnung=zeros(1,num_mfcc);
                        a=0;
                        for k=1:num_frames
                            if (c(k)==x)
                                rechnung= V1(:,k)'+rechnung;
                                a=a+1;
                            end 
                        end
                        C(x,:)=rechnung/a;
                    end
                    %%Analisieren
                    dis_unterschied=distance/distance_alt;
                    distance_alt=distance;
                end

            end
            save([pwd , '\' ,get(handles.edit4,'String'),'\',deblank(char(name(u,:))),'_Codebook_',num2str(codebook_size),'.mat'],'C','c','name')
        end
        codebook_size=codebook_size*2;    
    end
end

if get(handles.radiobutton6,'Value')==1
    codebook_size=1;
    i=1;
    while i<=str2num(get(handles.edit7,'String'))
        for u=1:setper
            clearvars -except u i VX name setper handles
            Nummer1='0';
            Was='S';
            thr=1.00005;
            V1=cell2mat(VX(u));
            loop=0;
            [num_mfcc num_frames] =size(V1);
            Var=zeros(num_mfcc,num_mfcc*i);
            M(1,:)=sum(V1')/num_frames;
            for x=1:i
                M(x,:)=M(1,:)+0.5*rand(1,size(V1,1));
            end
            for x=1:i
                for k=1:num_frames
                Var(:,num_mfcc*(x-1)+1:x*num_mfcc)=(V1(:,k)-M(x,:)')*(V1(:,k)-M(x,:)')'+ Var(:,num_mfcc*(x-1)+1:x*num_mfcc);
            end
            end
            Var=Var/num_frames;
            for x=1:i
                gew(x)=1/i;
            end
            while 1
            W_ges=0;
            %Inverse und det Berechnen
            for x=1:i
                V=Var(:,((x-1)*num_mfcc)+1:x*num_mfcc);
                d_V(x)=det(V);
                if d_V(x)<10^(-14)
                    break
                end
                i_V(:,((x-1)*num_mfcc)+1:x*num_mfcc)=inv(V);
            end
            if d_V(x)<10^(-14)
                break 
            end
            %Wahscheinlmichkeinten
            for k=1:num_frames
                W_f=0;
                for x=1:i
                    i_V_f=i_V(:,((x-1)*num_mfcc)+1:x*num_mfcc);
                    W_f=gew(x)*(1/((2*pi)^(num_mfcc/2)*sqrt(d_V(x)))*exp(-0.5*(V1(:,k)-M(x,:)')'*i_V_f*(V1(:,k)-M(x,:)')))+W_f;
                end
                W_ges=log(W_f)+W_ges;
                for j=1:i
                    i_V_f=i_V(:,((j-1)*num_mfcc)+1:j*num_mfcc);
                    W(k,j)=gew(j)*(1/((2*pi)^(num_mfcc/2)*sqrt(d_V(j)))*exp(-0.5*(V1(:,k)-M(j,:)')'*i_V_f*(V1(:,k)-M(j,:)')))/W_f;
                end
            end
            if exist ('W_ges_alt')
                vergleich=W_ges_alt/W_ges;
                if vergleich<thr 
                    break
            end   
            end
            %Vergleich
            W_ges_alt=W_ges;
            loop= loop+1;      
            W_ges_l(loop)=W_ges;
            M_bef=M;
            g_bef=gew;
            Var_bef=Var;
            %Mittelwert neu
            for x=1:i
                Mx=0;
                for k=1:num_frames  
                    Mx=W(k,x)*V1(:,k)+Mx;
                end
                M(x,:)=Mx'/sum(W(:,x));
            end
            %Varianze neu
            for x=1:i
                Varx=0;
                for k=1:num_frames  
                    Varx=W(k,x)*(V1(:,k)-M(x,:)')*(V1(:,k)-M(x,:)')'+Varx;
                end
                Var(:,((x-1)*num_mfcc)+1:x*num_mfcc)=Varx/sum(W(:,x));
                for l=1:num_mfcc
                    if Var(l,l+(num_mfcc*(x-1)))<0.1
                        Var(l,l+(num_mfcc*(x-1)))=0.1;
                    end
                end
            end
            %Gewichtung neu
            for x=1:i
                gew(x)=sum(W(:,x))/num_frames;
            end
            end
            save([pwd , '\' ,get(handles.edit4,'String'),'\',deblank(char(name(u,:))) ,'_GMM_',num2str(i),'.mat'],'M_bef','Var_bef','g_bef','name')
        end
        i=i*2;
    end
end  



if get(handles.knnbpbutton3,'Value')==1
    grenze=7;
    for x=1:setper
        prueba=(cell2mat(VX(x)))/(grenze);
        for z=1:size(prueba,2)
            for y=1:size(prueba,1)
                if prueba(y,z)>1
                prueba(y,z)=1;
                end
                 if prueba(y,z)<-1
                    prueba(y,z)=-1;
                 end
            end
        end
         f(x)={prueba};
    end
   set(handles.text16,'String','0%')
   drawnow
    hidden_layer=1;
    i2=0;
    hidden_layer_neurons= str2num(get(handles.edit5,'String'))
    eingang=11;
    ausgang=setper;
    schritte=0.05;
    for x=1:hidden_layer-1
        W(:,:,x)=-1+2*rand(hidden_layer_neurons,hidden_layer_neurons);
    end
    W_in=-1+2*rand(eingang,hidden_layer_neurons);
    W_intial=W_in;
    W_out=-1+2*rand(hidden_layer_neurons,ausgang);
    W_initial_ausgang=W_out;
    %Bias
    Bias_h=ones(hidden_layer_neurons,1);
    Bias_o=ones(ausgang,1);
    %Wunscht Wert
    gewuenscht=diag(ones(1,setper));
    %Anzahl von frames
    frames_eingang=eingang/11;
    for loop=1:2000000
        clear V1
        u =setper + ceil((-setper)*rand);
        V1=cell2mat(f(u));
        n =size(V1,2)-eingang+ ceil((-size(V1,2)+eingang)*rand);
        O_in(:,1)=V1(:,n);
        %Berechnung des Ausgang
        O(:,1)=W_in'*O_in+Bias_h;
        for z=1:size(O,1)
            O(z,1)=1/(1+exp(-O(z,1)));                          
        end
        O_out=W_out'*O(:,size(O,2))+Bias_o;
        for z=1:size(O_out,1)
            O_out(z,1)=1/(1+exp(-O_out(z,1)));
        end

        Error(loop)=1/2* sum((O_out-gewuenscht(:,u)).^2);
        %Gamma berechnen
        Gamma_out=O_out.*(1-O_out).*(O_out-gewuenscht(:,u));
        for x=size(O,2):-1:1
            if x==size(O,2)
                Gamma(:,x)=O(:,x).*(1-O(:,x)).*(W_out*Gamma_out);
            else
                Gamma(:,x)=O(:,x).*(1-O(:,x)).*(W(:,:,x)*Gamma(:,x+1));
            end
        end
        %Neue Werte der Gewichtung
        for j=1:size(W_in,1)
            for i=1:size(W_in,2)
                delta_W_in(j,i)=-schritte*Gamma(i,1)*O_in(j,1);
            end
        end
        W_in=delta_W_in+W_in;
        for j=1:size(Bias_h,1)
            delta_Bias_h(j,1)=-schritte*Bias_h(j,1)*Gamma(j,1);
        end
        Bias_h=delta_Bias_h+Bias_h;

        for j=1:size(W_out,1)
            for i=1:size(W_out,2)
                delta_W_out(j,i)=-schritte*Gamma_out(i)*O(j,size(O,2));
            end
        end
        W_out=delta_W_out+W_out;

        for j=1:size(Bias_o,1)
            delta_Bias_o(j,1)=-schritte*Bias_o(j,1)*Gamma_out(j,1);
        end
        Bias_o=delta_Bias_o+Bias_o;
        if mod(loop,20000)==0
            i2=i2+1;    
            set(handles.text16,'String',[num2str(i2),' %'])
            drawnow
        end
    end
    save([pwd , '\' ,get(handles.edit4,'String'),'\Neural_network_back_propagation_B_',num2str(hidden_layer_neurons),'_Final.mat'],'W_out','W_in','Bias_o','Bias_h','name');

end

if get(handles.radiobutton4,'Value')==1
output =4; 
set(handles.text16,'String','0%')
drawnow
hidden_layer=4;
i2=0;
output=str2num(get(handles.edit5,'String'))
    clearvars -except  VX name setper handles output i2
    
    %% Variable (einstellbar)
    input = 11;                               
    Gamma = 0.005;
    thr=10;
    if output<=10
        th_max=500;
    elseif output<=22 & output>10
        th_max=1000;
    elseif output<=50 & output>22   
        th_max=2000;
    elseif output<=200 & output>50   
        th_max=2500;
     elseif output<=550 & output>200    
        th_max=5000;
    end
    obere_grenze=7;
    untere_grenze=0;
    Wiederholungen=400;
    %---Amplitude
    v_pre = 0.6; 
    v_post_p = 2.9;
    v_post_n  = 2.3;
    v_th_init   = 2.0;     
    %----Zeit
    tges    = 7e-3;                                
    dt      = 0.5e-3;                              
    t_len   = tges/dt;   
    %% Pulse in Vektorform
    del_t_pre       = 12;                           
    del_t_post_p    = 6;                           
    del_t_post_n    = 6; 
    dt_pre      = del_t_pre * dt;                  
    del_t_post  = del_t_post_p + del_t_post_n;      
    dt_post_p   = del_t_post_p * dt;               
    dt_post_n   = del_t_post_n * dt;   
    untere_grenze=untere_grenze-obere_grenze;
    %----------------------------Input-Puls(pre pulse)-------------------------
    V_memr_pre  = zeros(t_len,1);                   
    for j = 2:(del_t_pre+1)
        V_memr_pre(j,1) = v_pre;
    end;
    %---------------------------Output-Puls (post pulse)-----------------------
    V_memr_post_puls    = zeros(t_len,1);           
    for j = 2 : (del_t_post+1)
        if j <= (del_t_post_p+1)
            V_memr_post_puls(j,1) = v_post_p; 
        else
            V_memr_post_puls(j,1) = -v_post_n;
        end
    end
    %---------------------------Anfangs einstellung---------------------------- 
    v_th_akt=ones(output,1)*v_th_init; 
    fire_neurons=zeros(output,1);
    V_memr_in=zeros(t_len,input);
    u_neur_o=zeros(t_len,output);
    t_0=zeros(1,output);
    V_memr_ges=zeros(t_len,input,output);
    gamma_c  = zeros(output,1); 
    W_Mem(:,:) = normrnd(0.5,0.1,input,output);
    W_mem_alt(:,:)=W_Mem(:,:);
    u_akt       = zeros(output,1); 
    zaehler=0;
    for w=1:Wiederholungen
        for u=1:setper
            muster_org = cell2mat(VX(u)) ; 
            for loop=1:size(muster_org,2)
                zaehler=zaehler+1;
                V_memr_post = zeros(t_len,output);
                spiked= 0;
                muster=muster_org(:,loop);
                %--------------------Eingang -> Stochatisch------------------------
                for mm = 1 : input
                 r =obere_grenze+ untere_grenze*rand;                                                          
                    for i = 1 : t_len
                        if r <= abs(muster(mm))
                            V_memr_in(i,mm) = V_memr_pre(i,1) * (muster(mm) / abs(muster(mm)));     
                        else
                            V_memr_in(i,mm) = 0;                                                   
                        end
                    end
                end
                %-------------Eingabe der Outputneuronen und Berechnung------------
                i_neur_in=V_memr_in*W_Mem(:,:);
                for nn = 1 : output                                                                                              
                        [uout, t_sp, u_akt_neu,ur] = neuron(i_neur_in(:, nn), t_len, dt, u_akt(nn), v_th_akt(nn,1));         
                        u_neur_o(:,nn)  = uout;                                                                             
                        u_akt(nn)       = u_akt_neu;                                                                        
                        t_0(nn)         = t_sp;                                                                             
                end
                %-----------------Analisieren ob es gefeuert wurde----------------- 
                t_first = min(t_0);
                if (t_first) > 0 && (t_first ~= 1000)                                 
                    c=find(t_0 == min(t_0(:)));
                    gamma_c(c)=gamma_c(c)+1;
                    fire_neurons(c)=fire_neurons(c)+1;
                    u_akt = ones(output,1) * ur;                                     
                    u_neur_o(t_first+1:t_len,:) = zeros(t_len-t_first,output);
                end
                if t_first < 1000 
                   spiked=1;     
                   for x=1:size(c,2)
                        V_memr_post(:,c(x)) = V_memr_post_puls;
                   end
                end
                %-------------------------Neue Gewichtung---------------------------
                if spiked == 1
                    for mm = 1 : input                                                      
                        for nn = 1 : output                                                                                                                       
                            V_memr_ges(:,mm,nn) = V_memr_in(:,mm) + V_memr_post(:,nn);      
                            for k = 1 : 2
                                if k==1
                                    V_in=V_memr_ges(2,mm,nn); 
                                    dt_temp=dt_post_p;                                      
                                end         
                                if k==2
                                    V_in=V_memr_ges(t_len-1,mm,nn); 
                                    dt_temp=dt_post_n;                                     
                                end        
                                [W_neu]      = memristor(V_in,dt_temp, W_Mem(mm,nn));      
                                W_Mem(mm,nn) = W_neu;                                       
                            end
                        end
                    end
                    clear c
                end
                %----------------------Einstellung von threshold----------------------
                if mod(zaehler,th_max)==0
                    for nn = 1 : output
                       v_th_akt(nn,1) = v_th_akt(nn,1) - v_th_akt(nn,1) * Gamma*(thr-gamma_c(nn));          
                    end
                    gamma_c  = zeros(output,1); 
                end 
            end
        end
        if mod(w,4)==0
            i2=i2+1;    
            set(handles.text16,'String',[num2str(i2),' %'])
            drawnow
        end
    end
    u_akt2=zeros(output,1);
    neuron_fire_wahrscheinlichkeit=zeros(setper,output);
    neuron_fire=zeros(setper,output);
    V_memr_in2=zeros(t_len,input);
    u_neur_o2=zeros(t_len,output);
    t_02=zeros(1,output);
    neuron_fire_wahrscheinlichkeit_W=zeros(setper,output);
    for Muster_numer=1:setper
        muster_org = cell2mat(VX(Muster_numer)) ; 
        neuron_gewinner=zeros(output,size(muster_org,2),setper);
        for loop2=1:size(muster_org,2)
            fire_neurons_test=zeros(1,output);
            loop_in=0;
            while loop_in<30
                loop_in=loop_in+1;
                muster=muster_org(:,loop2);
                %-------------------------Eingang Stochatisch----------------------
                for mm = 1 : input
                 r =obere_grenze+ (untere_grenze)*rand;                                                          
                    for i = 1 : t_len
                        if r <= abs(muster(mm))
                            V_memr_in2(i,mm) = V_memr_pre(i,1) * (muster(mm) / abs(muster(mm)));     
                        else
                            V_memr_in2(i,mm) = 0;                                                   
                        end
                    end
                end
                %---------------Eingabe der Outputneuronen und Berechnung----------
                i_neur_in2=V_memr_in2*W_Mem(:,:);
                for nn = 1 : output                                                                                              
                        [uout, t_sp, u_akt_neu,ur] = neuron(i_neur_in2(:, nn), t_len, dt, u_akt2(nn), v_th_akt(nn,1));         
                        u_neur_o2(:,nn)  = uout;                                                                             
                        u_akt2(nn) = u_akt_neu;                                                                        
                        t_02(nn) = t_sp;     
                end
                t_first2 = min(t_02);
                %-------------------Suchen von Neuron das gefeuert hat------------
                if (t_first2) > 0 && (t_first2 ~= 1000)                                 
                    pr=t_02;
                    c2=find(pr == min(pr));
                    fire_neurons_test(c2)=fire_neurons_test(c2)+1;
                    u_akt2 = ones(output,1) * ur;                                     
                    u_neur_o2(t_first+1:t_len,:) = zeros(t_len-t_first,output);
                end
            end
            %--------------Suchen von neuron was am meisten gefeuert hat----------

            if max(fire_neurons_test)>0
                x=find(fire_neurons_test == max(fire_neurons_test));
                for n=1:size(x,2)
                    neuron_gewinner(n,loop2,Muster_numer)=x(n);
                end
            else
                neuron_gewinner(1,loop2,Muster_numer)=0;
            end
    end
        neuron_gewinner_gesamt=zeros(output,1);
        for x=1:size(neuron_gewinner,2)
            for y=1:size(neuron_gewinner,1)
                if neuron_gewinner(y,x,Muster_numer)~=0
                    neuron_gewinner_gesamt(neuron_gewinner(y,x,Muster_numer))=neuron_gewinner_gesamt(neuron_gewinner(y,x,Muster_numer),1)+1;
                end
            end
        end
        wahrscheinlichkeit=neuron_gewinner_gesamt/sum(neuron_gewinner_gesamt)*100;
        for x=1:size(neuron_gewinner_gesamt,1)
            max_neuron_gewinner_gesamt=find(neuron_gewinner_gesamt == max(neuron_gewinner_gesamt));
            neuron_fire(Muster_numer,x)=max_neuron_gewinner_gesamt(1);        
            neuron_fire_wahrscheinlichkeit(Muster_numer,x)=wahrscheinlichkeit(max_neuron_gewinner_gesamt(1));
            neuron_gewinner_gesamt(neuron_fire(Muster_numer,x))=-1;

        end
    end
   % Erstelung von W_mem Methode 1
   for y=1:output
        wii=0;
        for x=1:size(neuron_fire,1)
            c(x)=find(neuron_fire(x,:)==y);    
               wii=neuron_fire_wahrscheinlichkeit(x,c(x))+wii;
        end
        for x=1:size(neuron_fire,1)
            if wii>0
                neuron_fire_wahrscheinlichkeit_W(x,c(x))=neuron_fire_wahrscheinlichkeit(x,c(x))/wii;   
            else
                neuron_fire_wahrscheinlichkeit_W(x,c(x))=0;
            end
        end
    end
    neuron_fire_tr=neuron_fire;  
    save([pwd , '\' ,get(handles.edit4,'String'),'\Hebb_Matrix_',num2str(output),'_Final.mat'],'W_Mem','v_th_akt','neuron_fire_wahrscheinlichkeit_W','neuron_fire_tr','V_memr_pre','obere_grenze','untere_grenze');
    set(handles.text16,'String','100 %')
    drawnow

end

function edit2_Callback(hObject, eventdata, handles)
% hObject    handle to edit2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit2 as text
%        str2double(get(hObject,'String')) returns contents of edit2 as a double


% --- Executes during object creation, after setting all properties.
function edit2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in pushbutton1.
function pushbutton1_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global setper name VX
setper=setper+1;
x=(get(handles.edit2,'String'));
name(setper,:)=[x,blanks(15-size(x,2))];
set(handles.SetPer,'String',char(name))
set(handles.text7,'String',setper)
%Recording
recObj = audiorecorder(44100, 24, 1);
disp('Start speaking.')
axes(handles.axes2)
matlabImage = imread([pwd ,'\images\player_record.png']);
image(matlabImage)
axis off
axis image
recordblocking(recObj, 8);
axes(handles.axes2)
matlabImage = imread([pwd ,'\images\player_record2.png']);
image(matlabImage)
axis off
axis image

disp('End of Recording.');
drums = getaudiodata(recObj);
drums = filter([1, -0.95], 1, drums);

fs=44100;
BP=10000;
Frame_number=4;
frame_duration=0.02;
mel_index=21;
DCT_index=12;
frame_duration_pausen=0.1;
thr=0.0025;
dctm = @( DCT_index, mel_index )( sqrt(2.0/mel_index) * cos( repmat([0:DCT_index-1].',1,mel_index).* repmat(pi*([1:mel_index]-0.5)/mel_index,DCT_index,1) ) );

%% Daten
drums_test=drums;%(1:fs*10);
drums_test_size=size(drums_test);
i=0;
frame_len_pausen=frame_duration_pausen*fs;
N=length(drums_test);
num_frames_pausen = floor(N/frame_len_pausen);
for k=2:num_frames_pausen
    frame = drums_test((k-1)*frame_len_pausen +1: frame_len_pausen*k);
    max_frame=max(frame);
    if (max_frame>thr) 
        i=i+1;
        drums_p((i-1)*frame_len_pausen +1: frame_len_pausen*i)=frame;
    end
end 
%drums_p=(drums_p/max(drums_p))*0.225;
sound(drums_p, 44100);
drums_p=drums_p';
frame_len=frame_duration*fs;
N=length(drums_p);
num_frames = floor(N/frame_len);
% Frames
for k=1:num_frames
    frame = drums_p((k-1)*frame_len +1: frame_len*k);
    v_MEL=zeros((BP*frame_len)/fs+1,1);
    %%Speech-pause entfernung
        %Window
        windowed= frame.*hamming(frame_len);
        %OneSideFFT
        Y=fft(windowed);
        P2 =abs(Y/frame_len).^2;
        P1 = P2(1:floor(frame_len/2)+1);           
        P1(2:end-1) = 2*P1(2:end-1);
        %Bandpass
        PB=P1(1:(BP*frame_len)/fs+1);       
        % Mel Filter
        mel_len = 2595*log10(1+BP/700)/mel_index;
        for x=1:mel_index-1 
            frq_mel_o = (10^((mel_len*(x+1))/2595)-1)*700;
            frq_mel_u = (10^((mel_len*(x-1))/2595)-1)*700;
            frq_mel_len=round((frq_mel_o-frq_mel_u)*frame_len/fs);
            P1X=P1(round(frq_mel_u*frame_len/fs)+1:round(frq_mel_o*frame_len/fs));
            m=size(P1X,1);
            mel_fil=P1X.*triang(m)./sum(triang(m));
            %op(x)=m;
            Dreiecke(:,x)=[frq_mel_u frq_mel_o];
            E_m(x)=log(sum(mel_fil));
            v_MEL(round(frq_mel_u*frame_len/fs)+1:round(frq_mel_o*frame_len/fs))=mel_fil+v_MEL(round(frq_mel_u*frame_len/fs)+1:round(frq_mel_o*frame_len/fs));    
        end
        % DCT
        %Alternative (1)
        C_DCT = dct(E_m,DCT_index);
        MEL(:,k)=E_m;
        M_MEL(:,k)=v_MEL;
        new_drums((k-1)*frame_len +1: frame_len*k)=frame;
        M_Y(:,k)=P1;
        M_y(:,k)=windowed;
        P(:,k)=PB;
        DCT_M(:,k)=C_DCT;
end
new_drums_ge= new_drums(1:frame_len*i);


%%Plots
f = fs*(0:(BP*frame_len)/fs)/frame_len;
axes(handles.axes3);
ti=frame_duration*(0:(k-1));
image(ti,f,log(P),'CDataMapping','scaled')
set(gca,'Ydir','Normal')
colorbar
title('Spektrogramm')
xlabel('Zeit in s')
ylabel('Frequenz in Hz')
DCT_M3=dctm(DCT_index, mel_index-1)*MEL;
DCT_M3=DCT_M3(2:12,:);
y = [1 DCT_index];
axes(handles.axes4);
imagesc(ti,y,DCT_M3,[-5 15])
set(gca,'Ydir','Normal')
colorbar
title('Energie der MFCC')
xlabel('Zeit in s')
ylabel('Cepstral Coefficients')
VX(setper,1)={DCT_M3};


%}
% --- Executes on button press in pushbutton2.
function pushbutton2_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global VX name setper
mkdir([pwd , '\' ,get(handles.edit4,'String')])
save([pwd , '\' ,get(handles.edit4,'String'),'\Allgemein.mat'],'VX','name','setper')


function edit3_Callback(hObject, eventdata, handles)
% hObject    handle to edit2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit2 as text
%        str2double(get(hObject,'String')) returns contents of edit2 as a double


% --- Executes during object creation, after setting all properties.
function edit3_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



% --- Executes on button press in pushbutton6.
function pushbutton6_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes during object creation, after setting all properties.
function axes2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to axes2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: place code in OpeningFcn to populate axes2


% --- Executes during object creation, after setting all properties.
function axes1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to axes1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: place code in OpeningFcn to populate axes1


% --- Executes during object creation, after setting all properties.
function axes7_CreateFcn(hObject, eventdata, handles)
% hObject    handle to axes1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: place code in OpeningFcn to populate axes1


% --- Executes on button press in pushbutton5.
function pushbutton5_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)



function edit4_Callback(hObject, eventdata, handles)
% hObject    handle to edit4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit4 as text
%        str2double(get(hObject,'String')) returns contents of edit4 as a double


% --- Executes during object creation, after setting all properties.
function edit4_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in radiobutton5.
function radiobutton5_Callback(hObject, eventdata, handles)
% hObject    handle to radiobutton5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of radiobutton5


% --- Executes on button press in radiobutton6.
function radiobutton6_Callback(hObject, eventdata, handles)
% hObject    handle to radiobutton6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of radiobutton6



function edit7_Callback(hObject, eventdata, handles)
% hObject    handle to edit7 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit7 as text
%        str2double(get(hObject,'String')) returns contents of edit7 as a double


% --- Executes during object creation, after setting all properties.
function edit7_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit7 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit6_Callback(hObject, eventdata, handles)
% hObject    handle to edit6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit6 as text
%        str2double(get(hObject,'String')) returns contents of edit6 as a double


% --- Executes during object creation, after setting all properties.
function edit6_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit5_Callback(hObject, eventdata, handles)
% hObject    handle to edit5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit5 as text
%        str2double(get(hObject,'String')) returns contents of edit5 as a double


% --- Executes during object creation, after setting all properties.
function edit5_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
