
function format_plural(number, unit)
{
    if (number != 1)
    {
        unit = unit + 's';
    }
    return `<span style="font-size:25px">${number}</span> <span style="font-size:15px">${unit}</span>`
}

function format_seconds_long(t_seconds)
{
    var days = Math.floor(t_seconds / (3600 * 24));
    var hours = Math.floor((t_seconds % (3600 * 24)) / 3600);
    var minutes = Math.floor((t_seconds % (3600 * 24) % 3600) / 60);
    var seconds = Math.floor(((t_seconds % (3600 * 24)) % 3600) % 60);

    d_f = (days != 0) ? format_plural(days, 'day') : "";

    d_f = (days != 0) ? format_plural(days, 'day') : "";
    h_f = (hours != 0) ? format_plural(hours, 'hour') : "";
    m_f = (minutes != 0) ? format_plural(minutes,'minute') : "";
    s_f = (seconds != 0) ? format_plural(seconds, 'second') : "";
    return `${d_f} ${h_f} ${m_f} ${s_f}`
}
function format_seconds_short(t_seconds)
{
    var days = Math.floor(t_seconds / (3600 * 24));
    var hours = Math.floor((t_seconds % (3600 * 24)) / 3600);
    var minutes = Math.floor((t_seconds % (3600 * 24) % 3600) / 60);
    var seconds = Math.floor(((t_seconds % (3600 * 24)) % 3600) % 60);

    d_f = (days != 0) ? format_plural(days, 'd') : "";
    h_f = (hours != 0) ? format_plural(hours, 'h') : "";
    m_f = (minutes != 0) ? format_plural(minutes,'m') : "";
    s_f = (seconds != 0) ? format_plural(seconds, 's') : "";
    return `${d_f} ${h_f} ${m_f} ${s_f}`
}

function progress_chart_track(inner_radius, outer_radius, color)
{
    return {outerRadius: outer_radius,
            innerRadius: inner_radius,
            backgroundColor: color,
            borderWidth: 0}
}

var training_progress_chart;

$(document).ready(
function (){
training_progress_chart = new Highcharts.Chart({
    chart: {
        renderTo:'progress-meter',
        type: 'solidgauge',
        margin: 0,
        padding: 0,
        horizontalAlign:'center'
    },

    title: {text: ''},
    tooltip: { enabled: false },

    pane: {
        startAngle: 0,
        endAngle: 360,
              background: [{ // Track for Move
                    outerRadius: '100%',
                    innerRadius: '90%',
                    backgroundColor: Highcharts.Color(Highcharts.getOptions().colors[1])
                        .setOpacity(0.3)
                        .get(),
                    borderWidth: 0
                }, { // Track for Exercise
                    outerRadius: '85%',
                    innerRadius: '75%',
                    backgroundColor: Highcharts.Color(Highcharts.getOptions().colors[1])
                        .setOpacity(0.3)
                        .get(),
                    borderWidth: 0
                }]},

    yAxis: {
        min: 0,
        max: 100,
        lineWidth: 0,
        tickPositions: []
    },

    plotOptions: {
        solidgauge: {
            dataLabels: {
                enabled: false
            },
            linecap: 'square',
            stickyTracking: false,
            rounded: false
        }
    },

    series: [{
        name: 'Move',
        data: [{
            color: Highcharts.getOptions().colors[9],
            radius: '100%',
            innerRadius: '90%',
            y: 0
        }]
    }, {
        name: 'Exercise',
        data: [{
            color: Highcharts.getOptions().colors[1],
            radius: '85%',
            innerRadius: '75%',
            y: 0
        }]
    }]
}
);})



function update_stats()
{
    $.getJSON('http://localhost:5000/json/training_stats.json',
            function(data) {
                //console.log(data);
                $('#stats-training-loss').text(data.training.loss.toFixed(2));
                $('#stats-training-accuracy').text((100*data.training.accuracy).toFixed(2));
                $('#stats-validation-loss').text(data.validation.loss.toFixed(2));
                $('#stats-validation-accuracy').text((100*data.validation.accuracy).toFixed(2));
            }
        );
}

function update_progress()
{
    $.getJSON('http://localhost:5000/json/training_progress.json',
            function(data) {
                $('#stats-batch-time').html(format_seconds_short(data.batch_time.toFixed(2)));
                $('#stats-epoch-time').html(format_seconds_short(data.epoch_time.toFixed(2)));
                $('#elapsed-time').html(format_seconds_long(data.elapsed_time.toFixed(2)));
                $('#remaining-time').html(format_seconds_long(data.remaining_time.toFixed(2)));
                $('#epoch-number').text(data.epoch_number);
                $('#total-epochs').text(data.total_epochs);
                training_progress_chart.series[1].points[0].update(data.percent_epoch_complete);
                training_progress_chart.series[0].points[0].update(data.percent_training_complete);
            }
        );

}
