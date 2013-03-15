function [tr, W] = stft(x, w, h)
    if nargin < 3
        h = 1;
    end
    if nargin < 2
        w = 256;
    end
    outlen = floor((length(x)-w) / h);
    W = hamming(w);
    tr = zeros(outlen, w);
    for i = 1:h:outlen
        tr(i,:) = fftshift(fft(W' .* x(i:i+w-1)));
    end
end

function window = hamming(l)
    n = l-1;
    if n == 0
        window = [];
        return;
    elseif n == 1
        window = 1;
        return
    end
    x = 0:n;
    window = .54-.46*cos(2*pi*x/n);
end