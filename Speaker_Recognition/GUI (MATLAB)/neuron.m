function [u_out, t_sp, u_akt_neu,ur] = neuron(I_in, tges_in, dt_in, u_akt, v_th_in) 
    
%% - Zeitparameter-------------------------------------------------------------------------------------//
t_length=tges_in;       % Gesamtzeit pro Intervall in Zeitschritten
dt=dt_in;               % Größe der Zeitschritte, [s]

%%- Bauteilparameter----------------------------------------------------------------------------------//
C = 5e-3;               % Kapazität des Models, [F]
R =5;                   % Widerstand des Models, [Ohm]
ur = -3;                        % Resetpotetntial
u0 = 0;                         % Ruhepotential
th = v_th_in;                   % Schwellwert
taum = R*C;                     % Zeitkosntante für Leckstrom

%% - Weitere Parameter---------------------------------------------------------------------------------//
u = zeros(t_length, 1);             % Erstellung Vektor für Membranpotential
u_out=zeros(t_length,1);            % Erstelleung Vektor für Spikes
t_sp=0;                             % Zeitpunkt des Spikes, 0 bedeutet: bisher wurde nicht gefeuert


for i=1:t_length;
    i_akt=I_in(i);
    u(i) =u_akt * exp(- ((dt) / taum)) +(1/C *i_akt)*dt;
    u_akt=u(i);
    t(i)=i;

    if(u(i) >= th)             %generate spike  
        u(i) = ur;
        u_out(i)=1;           
        u_akt_neu=ur;
        if t_sp == 0
            t_sp=i;
        end
    else
        u_out(i)=0;
    end
end

if t_sp == 0
    t_sp=1000;
end

u_akt_neu=u(t_length);

end 