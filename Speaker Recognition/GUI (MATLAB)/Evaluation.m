function varargout = Evaluation(varargin)
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @Evaluation_OpeningFcn, ...
                   'gui_OutputFcn',  @Evaluation_OutputFcn, ...
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


% --- Executes just before Evaluation is made visible.
function Evaluation_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to Evaluation (see VARARGIN)
axes(handles.axes4)
matlabImage = imread([pwd ,'\images\player_record2.png']);
image(matlabImage)
axis off
axis image
axes(handles.axes5)
matlabImage = imread([pwd ,'\images\Bild1.png']);
image(matlabImage)
axis off
axis image
axes(handles.axes6)
matlabImage = imread([pwd ,'\images\Bild2.png']);
image(matlabImage)
axis off
axis image


% Choose default command line output for Evaluation
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes Evaluation wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = Evaluation_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;













% --- Executes on button press in pushbutton1.
function pushbutton1_Callback(hObject, eventdata, handles)
recObj = audiorecorder(44100, 24, 1);
disp('Start speaking.')
axes(handles.axes4)
matlabImage = imread([pwd ,'\images\player_record.png']);
image(matlabImage)
axis off
axis image
recordblocking(recObj, get(handles.popupmenu7,'Value'));
axes(handles.axes4)
matlabImage = imread([pwd ,'\images\player_record2.png']);
image(matlabImage)
axis off
axis image

disp('End of Recording.');
drums = getaudiodata(recObj);
drums = filter([1, -0.95], 1, drums);

%% Variables
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
        
        %Alternative (3)
        %for a=1:DCT_index
        %   for b=1:mel_index-1
        %        C_DCT(a)=cos((a-1)*((b-1)-0.5)*pi/mel_index)*E_m(b);
        %   end
        %end
        %%Matrix erstellung
        MEL(:,k)=E_m;
        M_MEL(:,k)=v_MEL;
        new_drums((k-1)*frame_len +1: frame_len*k)=frame;
        M_Y(:,k)=P1;
        M_y(:,k)=windowed;
        P(:,k)=PB;
        DCT_M(:,k)=C_DCT;
        %OP(i,:)=prueba
end
new_drums_ge= new_drums(1:frame_len*i);


%%Plots
f = fs*(0:(BP*frame_len)/fs)/frame_len;
axes(handles.axes1);
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
axes(handles.axes2);
imagesc(ti,y,DCT_M3,[-5 15])
set(gca,'Ydir','Normal')
colorbar
title('Energie der MFCC')
xlabel('Zeit in s')
ylabel('Cepstral Coefficients')




V1=DCT_M3;

%% Personen-Set
X1=load([pwd ,'\', get(handles.edit1,'String'),'\Allgemein.mat'],'name','setper');
name=X1.name;
name=char(name);
User=X1.setper;
%User=10;
%name= ['Gabriel   '; 'Daniel    '; 'Christin  ';'Thomas    ';'Katharina ';'Martin    ';'Sana      '; 'Marina    ';'Konstantin'; 'Uwe       '];


Nummer1=get(handles.popupmenu6,'String');
Art=get(handles.popupmenu1,'Value');

%% Rechnung
distance=zeros(1,User);
num_mfcc=size(V1,1);




%% ---------------------Codebooks-------------------------------------------------------------
if  Art==1;
    for x=1:User
        X1=load([pwd ,'\',get(handles.edit1,'String'),'\',deblank(char(name(x,:))) ,'_Codebook_',num2str(Nummer1),'.mat'],'C');
        C(:,:,x)=X1.C;
        num_code(x)=size(C,1);
    end
    % Loop
    for k=1:size(V1,2)
     Zahl=0;
     for u=1:User
        for x=1:num_code(u)
            d(x)= sum(abs(V1(:,k)'-C(x,:,u)));
        end
      c(u)=find(d == min(d(:)));
      distance(u)= sum(abs(V1(:,k)'-C(c(u),:,u)))+ distance(u);
     end
    end
    Gewinner=find(distance == min(distance(:)))
    %Vergleich = sort(distance(:));
    %Vergleich = Vergleich(2)-Vergleich(1) 
end



%% --------------------------------GMM------------------------------------------------------------
if Art==2
    for x=1:User
    X1=load([pwd ,'\',get(handles.edit1,'String'),'\',deblank(char(name(x,:))) ,'_GMM_',num2str(Nummer1),'.mat'],'g_bef','M_bef','Var_bef');
    M(:,:,x)=X1.M_bef;
    Var(:,:,x)=X1.Var_bef;
    gew(:,:,x)=X1.g_bef;
    end

    i=size(M,1);
    for z=1:User
    for x=1:i
        V=Var(:,((x-1)*num_mfcc)+1:x*num_mfcc,z);
        d_V(x,z)=det(V);
        i_V(:,((x-1)*num_mfcc)+1:x*num_mfcc,z)=inv(V);
    end
    end
    for k=1:size(V1,2)
        W_f=0;
        for z=1:User
        for x=1:i
            i_V_f=i_V(:,((x-1)*num_mfcc)+1:x*num_mfcc,z);
            W_f=gew(1,x,z)*(1/((2*pi)^(num_mfcc/2)*sqrt(d_V(x,z)))*exp(-0.5*(V1(:,k)-M(x,:,z)')'*i_V_f*(V1(:,k)-M(x,:,z)')))+W_f;
        end
        end
        for z=1:User
        for j=1:i
            i_V_f=i_V(:,((j-1)*num_mfcc)+1:j*num_mfcc,z);
            W(k,j,z)=gew(1,j,z)*(1/((2*pi)^(num_mfcc/2)*sqrt(d_V(j,z)))*exp(-0.5*(V1(:,k)-M(j,:,z)')'*i_V_f*(V1(:,k)-M(j,:,z)')))/W_f;
        end
        W_g(k,z)=sum(W(k,:,z));
        end
    end
    clear max
    W_g=sum(W_g)/size(V1,2);
    disp([num2str(100*max(W_g)),'% wahscheinlichkeit'])
    Gewinner=find(W_g == max(W_g(:)))
end

%% Backpropagation
if Art==3
    grenze=7;
    Wx=load([pwd ,'\',get(handles.edit1,'String'),'\Neural_network_back_propagation_B_',num2str(Nummer1),'_Final.mat'],'W_out','W_in','W_intial','W_initial_ausgang','min_n','max_n','Bias_o','Bias_h');
    W_out=Wx.W_out;
    W_in=Wx.W_in;
    Bias_h=Wx.Bias_h;
    Bias_o=Wx.Bias_o;
    hidden_layer_neurons=size(W_in,2);
    eingang=size(W_in,1);
    ausgang=size(W_out,2);
    Gewinner=zeros(1,ausgang);
    O_in=zeros(eingang,1);
    count=zeros(1,User);
    O_out_g=zeros(ausgang,1);
    if isfield(Wx,'W')>0
        W=Wx.W;
        hidden_layer=size(W,3)+1;
    else
        hidden_layer=1;
    end
    V1=V1/grenze;
    for z=1:size(V1,2)
        for x=1:size(V1,1)
            if V1(x,z)>1
                V1(x,z)=1;
            end
             if V1(x,z)<-1
                V1(x,z)=-1;
            end
        end
    end
    for n=1:size(V1,2)  
        O=zeros(hidden_layer_neurons,1); 
        O_out=zeros(ausgang,1);  
        O_in(:,1)=V1(:,n);
        %% Berechnung des Ausgang
        O(:,1)=W_in'*O_in+Bias_h;
        for z=1:size(O,1)
            O(z,1)=1/(1+exp(-O(z,1)));                          
        end
        for x=1:hidden_layer-1
            O(:,x+1)=W(:,:,x)'*O(:,x);
            for z=1:size(O,1)
                O(z,x+1)=1/(1+exp(-O(z,x+1)));
            end
        end
        O_out=W_out'*O(:,size(O,2))+Bias_o;
        for z=1:size(O_out,1)
            O_out(z,1)=1/(1+exp(-O_out(z,1)));
        end
        O_out_g=O_out+O_out_g;
    end
    Gewinner=find(O_out_g == max(O_out_g(:)))
end


%% Hebb
if Art==4
    %name= ['Gabriel   ';'Christin  '; 'Daniel    ';'Uwe       ';'Thomas    ';'Katharina ';'Martin    ';'Sana      '; 'Marina    ';'Konstantin'];
    %X1=load(['C:\Users\Gabriel\Desktop\Master\Daten\',get(handles.edit1,'String'),'\Hebb_Matrix_',num2str(Nummer1),'_',num2str(User),'Final.mat'],'W_Mem','v_th_akt','neuron_fire_wahrscheinlichkeit_W','neuron_fire_tr','V_memr_pre','obere_grenze','untere_grenze');
    X1=load([pwd ,'\',get(handles.edit1,'String'),'\Hebb_Matrix_',num2str(Nummer1),'_Final.mat'],'W_Mem','v_th_akt','neuron_fire_wahrscheinlichkeit_W','neuron_fire_tr','V_memr_pre','obere_grenze','untere_grenze');
    neuron_fire_wahrscheinlichkeit_W=X1.neuron_fire_wahrscheinlichkeit_W;
    neuron_fire_tr=X1.neuron_fire_tr;
    W_Mem=X1.W_Mem;
    v_th_akt=X1.v_th_akt;
    V_memr_pre=X1.V_memr_pre;
    obere_grenze=X1.obere_grenze;
    untere_grenze=X1.untere_grenze;
    input=size(W_Mem,1);
    output=size(W_Mem,2);
    tges    = 7e-3;                                
    dt      = 0.5e-3; 
    t_len   = tges/dt;  
    u_akt=zeros(output,1);   
    neuron_fire=zeros(User,output);
    neuron_gewinner=zeros(output,size(V1,2),User);
    neuron_fire_Anzahl=zeros(User,output);
    for loop=1:size(V1,2) 
        fire_neurons_test=zeros(1,output);
        muster=V1(:,loop);
        for loop_in=1:10
            %-------------------------Eingang Stochatisch----------------------
            for mm = 1 : input
             r =obere_grenze+ (untere_grenze)*rand;                                                          
                for i = 1 : t_len
                    if r <= abs(muster(mm))
                        V_memr_in(i,mm) = V_memr_pre(i,1) * (muster(mm) / abs(muster(mm)));     
                    else
                        V_memr_in(i,mm) = 0;                                                   
                    end
                end
            end
            %---------------Eingabe der Outputneuronen und Berechnung----------
            i_neur_in=V_memr_in*W_Mem(:,:);
            for nn = 1 : output                                                                                              
                    [uout, t_sp, u_akt_neu,ur] = neuron(i_neur_in(:, nn), t_len, dt, u_akt(nn), v_th_akt(nn,1));         
                    u_neur_o(:,nn)  = uout;                                                                             
                    u_akt(nn) = u_akt_neu;                                                                        
                    t_0(nn) = t_sp;     
            end
            t_first = min(t_0);
            %-------------------Suchen von Neuron das gefeuert hat------------
            if (t_first) > 0 && (t_first ~= 1000)                                 
                pr=t_0;
                c=find(pr == min(pr));
                fire_neurons_test(c)=fire_neurons_test(c)+1;
                u_akt = ones(output,1) * ur;                                     
                u_neur_o(t_first+1:t_len,:) = zeros(t_len-t_first,output);
            end
        end
        %--------------Suchen von neuron was am meisten gefeuert hat----------

        if max(fire_neurons_test)>0
            x=find(fire_neurons_test == max(fire_neurons_test));
            for n=1:size(x,2)
                neuron_gewinner(n,loop)=x(n);
            end
        else
            neuron_gewinner(1,loop)=0;
        end
    end
    neuron_gewinner_gesamt=zeros(output,1);
    if sum(sum(neuron_gewinner(:,:)))>0
        for x=1:size(V1,2) 
            for y=1:output
                if neuron_gewinner(y,x)~=0
                    neuron_gewinner_gesamt(neuron_gewinner(y,x))=neuron_gewinner_gesamt(neuron_gewinner(y,x))+1;
                end
            end
        end
        Anzahl=neuron_gewinner_gesamt;
        for x=1:size(neuron_gewinner_gesamt,1)
            max_neuron_gewinner_gesamt=find(neuron_gewinner_gesamt == max(neuron_gewinner_gesamt));
            if Anzahl(max_neuron_gewinner_gesamt(1)) ~= 0
                neuron_fire(1,x)=max_neuron_gewinner_gesamt(1);        
                neuron_fire_Anzahl(1,x)=Anzahl(max_neuron_gewinner_gesamt(1));
                neuron_gewinner_gesamt(neuron_fire(1,x))=-1;
            else
                neuron_fire(1,x)=0;        
                neuron_fire_Anzahl(1,x)=0;
            end
        end
    else
        neuron_fire_Anzahl(1,:)=zeros(1,output);
        neuron_fire(1,:)=zeros(1,output);
    end
    Sprecher=zeros(size(neuron_fire,1),User);
    for x=1:size(neuron_fire,1)
        for y=1:size(neuron_fire,2)
            if neuron_fire(x,y)~= 0
                for z=1:User
                    num=find(neuron_fire_tr(z,:)==neuron_fire(x,y));
                    Sprecher(x,z)=neuron_fire_Anzahl(x,y)*neuron_fire_wahrscheinlichkeit_W(z,num)+Sprecher(x,z);
                end
            end
        end 
    end
    if sum(Sprecher(1,:))~=0
        Gewinner = find(Sprecher(1,:)==max(Sprecher(1,:)));
    else
        Gewinner=0;
    end
end





%% ------------------------------Gewinner------------------------------------------------
 set(handles.text2,'String',[deblank(char(name(Gewinner,:)))])
 set(handles.text6,'String',' hat am wahscheinlichsten gesprochen')
 
 
 
% --- Executes on selection change in popupmenu1.
function popupmenu1_Callback(hObject, eventdata, handles)
% hObject    handle to popupmenu1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
txt=get(handles.popupmenu1,'Value');
if txt==1
    set(handles.text4,'String','Vektoren')
end
if txt==2
    set(handles.text4,'String','Verteilungen')
end
if txt==3
    set(handles.text4,'String','Neuronen')
end
if txt==4
    set(handles.text4,'String','Neuronen')
end
% Hints: contents = cellstr(get(hObject,'String')) returns popupmenu1 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from popupmenu1


% --- Executes during object creation, after setting all properties.
function popupmenu1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to popupmenu1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in popupmenu6.
function popupmenu6_Callback(hObject, eventdata, handles)
% hObject    handle to popupmenu6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns popupmenu6 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from popupmenu6


% --- Executes during object creation, after setting all properties.
function popupmenu6_CreateFcn(hObject, eventdata, handles)
% hObject    handle to popupmenu6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes during object creation, after setting all properties.
function axes1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to axes1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: place code in OpeningFcn to populate axes1


% --- If Enable == 'on', executes on mouse press in 5 pixel border.
% --- Otherwise, executes on mouse press in 5 pixel border or over text3.
function text3_ButtonDownFcn(hObject, eventdata, handles)
% hObject    handle to text3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on key press with focus on popupmenu1 and none of its controls.
function popupmenu1_KeyPressFcn(hObject, eventdata, handles)
% hObject    handle to popupmenu1 (see GCBO)
% eventdata  structure with the following fields (see UICONTROL)
%	Key: name of the key that was pressed, in lower case
%	Character: character interpretation of the key(s) that was pressed
%	Modifier: name(s) of the modifier key(s) (i.e., control, shift) pressed
% handles    structure with handles and user data (see GUIDATA)



% --- If Enable == 'on', executes on mouse press in 5 pixel border.
% --- Otherwise, executes on mouse press in 5 pixel border or over popupmenu1.
function popupmenu1_ButtonDownFcn(hObject, eventdata, handles)
% hObject    handle to popupmenu1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on selection change in popupmenu7.
function popupmenu7_Callback(hObject, eventdata, handles)
% hObject    handle to popupmenu7 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns popupmenu7 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from popupmenu7


% --- Executes during object creation, after setting all properties.
function popupmenu7_CreateFcn(hObject, eventdata, handles)
% hObject    handle to popupmenu7 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit1_Callback(hObject, eventdata, handles)
% hObject    handle to edit1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit1 as text
%        str2double(get(hObject,'String')) returns contents of edit1 as a double


% --- Executes during object creation, after setting all properties.
function edit1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes during object creation, after setting all properties.
function figure1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to figure1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called
