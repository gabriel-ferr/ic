using JSON, Plots, BrainFlow, Statistics, DSP

json = JSON.parsefile("Data/DadosBrutos.json")
data = json["resultadosBrutos"]["Collector.Model.Tarefa"]

datetime_str_start = split(replace(data["tempoInicio"], "-03:00" => ""), 'T')
time_split_start = split(datetime_str_start[2], ':')

hour_start = parse(Int32, time_split_start[1])
minute_start = parse(Int32, time_split_start[2])
second_start = parse(Float32, time_split_start[3])

datetime_str_end = split(replace(data["tempoFim"], "-03:00" => ""), 'T')
time_split_end = split(datetime_str_end[2], ':')

hour_end = parse(Int32, time_split_end[1])
minute_end = parse(Int32, time_split_end[2])
second_end = parse(Float32, time_split_end[3])

# ------------------------------------
event_time = ((hour_end - hour_start) * 3600) + ((minute_end - minute_start) * 60) + (second_end - second_start)
# ------------------------------------
channels = []
for channel in data["dadosEEG"]
    content = parse.(Float64, replace.(split(channel, '_'), ',' => '.'))
    push!(channels, content)
end
# ------------------------------------
time = range(0, event_time, size(channels[1])[1])
# ------------------------------------
rate = 1 / (time[2] - time[1])
# ------------------------------------
lowpass = Filters.Lowpass(30; fs=rate)
highpass = Filters.Highpass(5; fs=rate)
notch = Filters.iirnotch(60, 30; fs=rate)

method = Filters.Butterworth(4)
# ------------------------------------
y_filter = []
for channel in channels
    y_lowpass = filt(digitalfilter(lowpass, method), channel)
    y_highpass = filt(digitalfilter(highpass, method), y_lowpass)
    push!(y_filter, filtfilt(notch, y_highpass))
end
# ------------------------------------
nfft = BrainFlow.get_nearest_power_of_two(Integer(250))
# ------------------------------------
#psd = BrainFlow.get_psd_welch(y_filter[1], nfft, Integer(nfft / 2), Integer(250), BrainFlow.BLACKMAN_HARRIS)
#band_power_alpha = BrainFlow.get_band_power(psd, 7.0, 13.0)
#band_power_beta = BrainFlow.get_band_power(psd, 14.0, 30.0)

#println(band_power_alpha / band_power_beta)

#plot(psd[2][:], psd[1][:])

#plot(time[:], y_filter[1][:])
#plot(time[:], y_filter[2][:])
#plot(time[:], y_filter[3][:])
#plot!(time[:], y_filter[4][:])
#plot!(time[:], y_filter[5][:])
#plot!(time[:], y_filter[6][:])
#plot!(time[:], y_filter[7][:])
#plot!(time[:], y_filter[8][:])
