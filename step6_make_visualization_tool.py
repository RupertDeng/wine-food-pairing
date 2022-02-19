from cmath import pi
from matplotlib import gridspec, pyplot


def make_nonaroma_radar_chart(grid, plot_index, data, color, plot_type):

  categories = ['sweet', 'acid', 'salt', 'piquant', 'fat', 'bitter']

  angles = [i / 6 * 2 * pi for i in range(6)]
  angles += angles[:1]

  ax = pyplot.subplot(grid[plot_index], polar=True)

  ax.set_theta_offset(pi / 2)
  ax.set_theta_direction(-1)

  pyplot.xticks(angles[:-1], categories, color='grey', size=11)

  ax.set_rlabel_position(0)
  pyplot.yticks([0, 0.25, 0.5, 0.75, 1.0, 1.25], ['0', '0,25', '0.5', '0.75', '1', '1.25'], color='grey', size=0)
  pyplot.ylim(-0.25, 1.25)

  values = [data[taste + ' scalar'] for taste in categories]
  values += values[:1]
  ax.plot(angles, values, color=color, linewidth=2, linestyle='solid')
  ax.fill(angles, values, color=color, alpha=0.4)

  if plot_type == 'wine':
    plot_title = data['Variety'] + '\n' + data['Geo'] + '\n' + '(' + data['pairing_type'] + ')'
  else:
    plot_title = 'Food Flavor Profile' '\n' + '(' + data['food'] + ')'

  pyplot.title(plot_title, size=13, color='black', y=1.2)


def make_weight_line(grid, line_index, data, dot_color):
  ax = pyplot.subplot(grid[line_index])
  ax.set_xlim(-1, 2)
  ax.set_ylim(0, 3)

  xmin = 0
  xmax = 1
  y = 1
  height = 0.2

  pyplot.hlines(y, xmin, xmax)
  pyplot.vlines(xmin, y-height/2, y+height/2)
  pyplot.vlines(xmax, y-height/2, y+height/2)

  px = data['weight scalar']
  pyplot.plot(px, y, 'ko', ms=10, mfc=dot_color)

  pyplot.text(xmin-0.1, y, 'Light-Bodied', horizontalalignment='right', fontsize=11, color='grey')
  pyplot.text(xmax+0.1, y, 'Full-Bodied', horizontalalignment='left', fontsize=11, color='grey')

  pyplot.axis('off')


def list_aroma_descriptors(grid, text_index, data):
  ax = pyplot.subplot(grid[text_index])
  ax.set_xticks([])
  ax.set_yticks([])
  for spine in ax.spines.values():
    spine.set_visible(False)
  ax.invert_yaxis()

  text = 'Complementary wine aromas: \n\n' + ', \n'.join(data['most_impactful_descriptor'])
  ax.text(x=0, y=1, s=text, fontsize=12, color='grey')


def plot_wine_recommendations(wine_data, food_data):
  subplot_rows = 5
  subplot_columns = 4
  pyplot.figure(figsize=(20, 10), dpi=96)

  grid = gridspec.GridSpec(subplot_rows, subplot_columns)
  food_radar_index, food_line_index = 0, 4
  wine_radar_index, wine_line_index, wine_text_index = 8, 12, 16

  make_nonaroma_radar_chart(grid, food_radar_index, food_data, 'orange', 'food')
  make_weight_line(grid, food_line_index, food_data, 'orange')

  for w in range(len(wine_data)):
    make_nonaroma_radar_chart(grid, wine_radar_index, wine_data[w], 'red', 'wine')
    make_weight_line(grid, wine_line_index, wine_data[w], 'red')
    list_aroma_descriptors(grid, wine_text_index, wine_data[w])
    wine_radar_index += 1
    wine_line_index += 1
    wine_text_index += 1
  
  pyplot.show()
    



