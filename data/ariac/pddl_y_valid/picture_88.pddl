(define (problem picture_88)
    (:domain ariac)
    
    (:objects
        table pump_placement regulator_placement battery_placement buffer_placement - location
        red_sensor green_sensor - part
    )
    
    (:init
        (robot_at table)
        (handempty)        
        (clear red_sensor)
        (clear green_sensor)
        (part_at red_sensor table)
        (part_at green_sensor table)
    )
    
    (:goal
        (and
            
        )
    )
)
