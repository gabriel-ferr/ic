#
#       Importa as bibliotecas que serão utilizadas no processo.
using JSON, Plots, BrainFlow, Statistics, DSP

#       Configurações.
#   Determina um pedding para o intervalo de dados.
start_pedding = 0
end_pedding = 0


#       Carrega os dados do arquivo JSON (os dados carregados são brutos).
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

#       Calcula o intervalo de tempo no qual o experimento foi realizado.
event_time = ((hour_end - hour_start) * 3600) + ((minute_end - minute_start) * 60) + (second_end - second_start)

#       Calcula um negócio que eu não sei pra que serve.
nfft = BrainFlow.get_nearest_power_of_two(Integer(250))

#       Separa os dados obtidos por canais.
#   Cada canal tem um Vector de Float32 contendo a amplitude do EEG.
channels = []
for channel in data["dadosEEG"]
    content = parse.(Float64, replace.(split(channel, '_'), ',' => '.'))
    push!(channels, content)
end

#       Cria um conjunto de dados de tempo.
#   Ele pega de referência o primeiro canal, com a hipótese deles terem o mesmo
#   tamanho.
data_size = size(channels[1])[1]
time = range(0, event_time, data_size)

#       Calcula o tamanho dos segmentos de dados que devem ser separados
#   para tratamento.
segment_size = trunc(Int32, (data_size / event_time))

#       Calcura a frequência base da coleta (normalmente entorno de 250 Hz)
rate = 1 / (time[2] - time[1])
println("Data Frequency: ", rate, "Hz")

#       Define os filtros
lowpass = Filters.Lowpass(30; fs=rate)
highpass = Filters.Highpass(5; fs=rate)
notch = Filters.iirnotch(60, 30; fs=rate)

method = Filters.Butterworth(4)

#       Aplica os filtros
y_filter = []
for channel in channels
    y_lowpass = filt(digitalfilter(lowpass, method), channel)
    y_highpass = filt(digitalfilter(highpass, method), y_lowpass)
    push!(y_filter, filtfilt(notch, y_highpass))
end

#       Separa os dados em cada canal por segmentos temporais.
#   Depois dessa execução os dados vão ser separados no seguinte formato:
#       Canal[1:8]
#           >> Segmentos de Tempo [1:(numb_of_segments + 1)]
#               >> Dados do segmento de Tempo no formato (tempo, amplitude) [1:size(egg_data[i][j])[1]]
numb_of_segments = trunc(Int32, event_time)
loss_elements = data_size - (numb_of_segments * segment_size)

for c = 1:length(y_filter)
    channel = y_filter[c]
    for i = 1:numb_of_segments
        graph = plot(time[(1 + ((i - 1) * segment_size)):(i * segment_size)], channel[(1 + ((i - 1) * segment_size)):(i * segment_size)])
        png(graph, "Output/Time/graph_" * string(c) * "_" * string(i))
    end

    if (loss_elements > 0)
        graph = plot(time[(1 + (data_size - loss_elements)):data_size], channel[(1 + (data_size - loss_elements)):data_size])
        png(graph, "Output/Time/graph_" * string(c) * "_" * string(numb_of_segments + 1))
    end
end

# Imagens ID 1 -> FP1
# Imagens ID 2 -> F4
# Imagens ID 3 -> F3
# Imagens ID 4 -> C4
# Imagens ID 5 -> FP2
# Imagens ID 6 -> O2
# Imagens ID 7 -> C3
# Imagens ID 8 -> O1

#       Separa a frequência usando a BrainFlow
for c = 1:length(y_filter)

    #       Calcula os valores.
    psd = BrainFlow.get_psd_welch(y_filter[c], nfft, Integer(nfft / 2), Integer(250), BrainFlow.BLACKMAN_HARRIS)

    graph = plot(psd[2][:], psd[1][:])
    png(graph, "Output/Frequency/graph_" * string(c))

    #       Separa valores relacionados as ondas Alpha e Beta,
    #   como eu não sei o significado exato vou chamar de "Quantidade".
    #   Extraí isso da biblioteca do BrainFlow.
    band_power_alpha = BrainFlow.get_band_power(psd, 7.0, 13.0)
    band_power_beta = BrainFlow.get_band_power(psd, 14.0, 30.0)

    println("Quantidade de ondas Alpha para o canal " * string(c) * ": ", band_power_alpha)
    println("Quantidade de ondas Beta para o canal " * string(c) * ": ", band_power_beta)

end