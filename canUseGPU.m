function tf = canUseGPU()
try
    tf = (gpuDeviceCount > 0);
catch
    tf = false;
end
end