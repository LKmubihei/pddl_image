(define (problem picture_91)
    (:domain ariac)
    
    (:objects
        table pump_placement regulator_placement battery_placement buffer_placement - location
        blue_sensor red_sensor green_sensor - part
    )
    
    (:init
        (robot_at table)
        (handempty)        
        (clear blue_sensor)
        (clear red_sensor)
        (part_at blue_sensor table)
        (part_at green_sensor table)
        (part_at red_sensor table)
    )
    
    (:goal
        (and
            
        )
    )
)
