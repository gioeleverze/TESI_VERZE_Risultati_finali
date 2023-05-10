function y = day_sliding_window(x, winSize, tau)
    y = [];
    for di = 1 : size(x, 1)
        [temp_samples, count] = sliding_window(x(di, :), winSize, tau);
        temp_samples = temp_samples(:,1:count);
        y = cat(2, y, temp_samples);
    end
end

