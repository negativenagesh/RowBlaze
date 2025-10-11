# Use the official Nginx image
FROM nginx:1.25-alpine

# Remove the default Nginx configuration
RUN rm /etc/nginx/conf.d/default.conf

# Copy our custom configuration into the container
COPY nginx/nginx.conf /etc/nginx/nginx.conf