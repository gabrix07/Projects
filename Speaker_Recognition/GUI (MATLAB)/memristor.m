function [w_neu] = memristor(delta_V, t_n, w_alt) 

% Einfügung der untere Grenze von 1e-5. Ansonsten würde die Funktion sehr lange brauchen, um von einem
% noch kleineren Zustandswert wieder auf einen hohen Wert zu gelangen.
if  w_alt < 1e-5 
    w_alt = 1e-5; 
end

% Nur wenn der Spannungsbetrag größer als 0.7 ist, wird eine Änderung des Zustandes bewirkt
if abs(delta_V) >= 0.7
    % Modell
    C=28.5e-3; 
 
    % (A) Fkt. dt
    alpha_t  = 1e-3 * exp(13*(t_n-8e-3));

    % (B) Fkt. dV
    alpha_dV = 1.51e-1*abs(delta_V);
    if delta_V <0
        alpha_dV = exp(5*(abs(delta_V)-2.38)); 
    end
 
    % (C) Plast. Modell
    beta_2 = 1/C * (alpha_dV) * w_alt;
    if delta_V >= 0 
        w_neu   = (w_alt+(1-w_alt) / (1+exp(-beta_2*alpha_t))) / 0.5 - 1; 
    else
        w_neu   = (w_alt-(w_alt) / (1+exp(-beta_2*alpha_t))) / 0.5;
    end 
else
    w_neu   = w_alt;   
end