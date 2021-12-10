import plotly.express as px

df = px.data.gapminder().query("year == 2008")
avg_lifeExp = (df['lifeExp']*df['pop']).sum()/df['pop'].sum()

print(px.colors.diverging.BrBG)

#fig = px.choropleth(df, locations="iso_alpha", color="lifeExp",
                    #color_continuous_scale=px.colors.diverging.BrBG,
                    #color_continuous_midpoint=avg_lifeExp,
                   # title="World Average Life Expectancy in 2007 in years was %.1f" % avg_lifeExp)
#fig.show()